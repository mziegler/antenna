"""
Import pre-computed Mothbox / Mothbot_Process detections + classifications into Antenna.

Mothbot_Process runs an external YOLO detector + a ``pybioclip`` classifier over raw
Mothbox frames and writes, for each raw image, a ``<image>_botdetection.json`` file plus
cropped insect "patch" images. This module ingests those results directly (bypassing
Antenna's live ML ``ProcessingService``), mapping them onto Antenna's
``Deployment → SourceImage → Detection → Classification → Occurrence → Taxon`` model.

Key mapping decisions (see the import plan for the full rationale):

- **Oriented bboxes → axis-aligned.** Antenna's ``Detection.bbox`` is an axis-aligned
  ``[x1, y1, x2, y2]`` in source-image pixels (top-left origin). Mothbox stores an oriented
  box as four corner ``points`` + a ``direction`` angle; we store the axis-aligned envelope
  of the four points and drop the angle (nothing in Antenna reads it).
- **Reuse the Mothbox patch crops.** Each detection's ``path`` is set to a full public URL
  of the existing patch in the data source's bucket, so no crop is recomputed and the patch
  becomes the occurrence-gallery image.
- **Deployment metadata comes from the JSON.** Every ``_botdetection.json`` carries top-level
  ``deployment_name``, ``project``, ``site``, ``latitude``, ``longitude``, ``device``, etc.,
  so deployments/projects are provisioned from the data itself.
- **Idempotent.** A source image that already has detections from this detector is skipped,
  so the same code serves the one-shot backfill and incremental per-night re-runs.
"""

import collections
import dataclasses
import json
import logging
import posixpath
import typing

from django.utils.text import slugify

from ami.main.models import (
    Classification,
    Deployment,
    Detection,
    Device,
    Project,
    S3StorageSource,
    Site,
    SourceImage,
    Taxon,
    TaxonRank,
)
from ami.ml.models import Algorithm, Pipeline
from ami.ml.models.algorithm import AlgorithmCategoryMap, AlgorithmTaskType

logger = logging.getLogger(__name__)

# Suffix Mothbot_Process gives the per-image detection JSON.
BOTDETECTION_SUFFIX = "_botdetection.json"
# Sub-folder (relative to a night folder) that holds the JSONs and patch crops.
PROCESSED_SUBDIR = "_processed"
# Label Mothbot_Process uses for a detected-but-unidentified insect (no taxonomy).
UNIDENTIFIED_LABEL = "creature"

# DarwinCore rank fields present on a Mothbox shape, ordered coarse → fine.
DWC_RANK_FIELDS: list[tuple[str, TaxonRank]] = [
    ("kingdom", TaxonRank.KINGDOM),
    ("phylum", TaxonRank.PHYLUM),
    ("class", TaxonRank.CLASS),
    ("order", TaxonRank.ORDER),
    ("family", TaxonRank.FAMILY),
    ("genus", TaxonRank.GENUS),
    ("species", TaxonRank.SPECIES),
]


@dataclasses.dataclass
class ImportSummary:
    """Counters returned by an import run (useful for job logs and tests)."""

    source_images_matched: int = 0
    source_images_skipped_no_capture: int = 0
    source_images_skipped_already_imported: int = 0
    detections_created: int = 0
    classifications_created: int = 0
    occurrences_created: int = 0
    json_files_read: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------
def bbox_from_points(points: typing.Sequence[typing.Sequence[float]]) -> list[float]:
    """Axis-aligned envelope ``[x1, y1, x2, y2]`` of an oriented box's corner points.

    Mothbox ``shape_type == "rotation"`` shapes store four ``[x, y]`` corners. Antenna has
    no rotation representation, so we keep the tight axis-aligned envelope; the ``direction``
    angle is intentionally discarded.
    """
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


# --------------------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------------------
def resolve_taxon(shape: dict, cache: dict[tuple[str, str], Taxon]) -> Taxon | None:
    """Resolve (or create) the most-specific Taxon for a shape's DarwinCore fields.

    Walks kingdom → species, ``get_or_create``-ing each populated rank and linking parents,
    and returns the deepest taxon (the determination). Returns ``None`` for an unidentified
    ``creature`` shape (no taxonomy fields). ``cache`` is keyed on ``(name, rank)`` and reused
    across a run so repeated orders don't re-query.

    Taxa are matched by name only, consistent with Antenna's ``import_taxa`` and
    ``AlgorithmCategoryMap.with_taxa`` — so pre-loading the GBIF species list with
    ``import_taxa`` gives these the correct parents/keys; anything missing is created bare here.
    """
    parent: Taxon | None = None
    deepest: Taxon | None = None
    for field, rank in DWC_RANK_FIELDS:
        name = (shape.get(field) or "").strip()
        if not name:
            continue
        key = (name, rank.value)
        taxon = cache.get(key)
        if taxon is None:
            taxon, created = Taxon.objects.get_or_create(name=name, defaults={"rank": rank.value})
            # Link the parent chain when we just created the node, or when an existing bare
            # taxon has no parent yet — but never overwrite an existing, more-authoritative parent.
            if parent is not None and taxon.parent_id is None and taxon.pk != parent.pk:
                taxon.parent = parent
                taxon.save()
            elif created:
                taxon.save()
            cache[key] = taxon
        parent = taxon
        deepest = taxon
    return deepest


# --------------------------------------------------------------------------------------
# Algorithms & pipeline
# --------------------------------------------------------------------------------------
def get_or_create_category_map(labels_with_rank: list[tuple[str, str]]) -> AlgorithmCategoryMap:
    """Build an ``AlgorithmCategoryMap`` from ``(label, taxon_rank)`` pairs.

    Labels match ``Taxon.name`` so ``with_taxa()`` can resolve them. Reused by label-set hash.
    """
    labels = [label for label, _ in labels_with_rank]
    data = [{"index": i, "label": label, "taxon_rank": rank} for i, (label, rank) in enumerate(labels_with_rank)]
    labels_hash = AlgorithmCategoryMap.make_labels_hash(labels)
    category_map = AlgorithmCategoryMap.objects.filter(labels_hash=labels_hash).first()
    if category_map is None:
        category_map = AlgorithmCategoryMap.objects.create(data=data, labels=labels, version="mothbox")
    return category_map


def get_or_create_algorithm(
    name: str,
    task_type: AlgorithmTaskType,
    category_map: AlgorithmCategoryMap | None = None,
) -> Algorithm:
    """``get_or_create`` an Algorithm by its slugified key, setting task type / category map."""
    key = slugify(name)
    algorithm, created = Algorithm.objects.get_or_create(
        key=key,
        defaults={"name": name, "task_type": task_type.value, "category_map": category_map},
    )
    if not created and category_map is not None and algorithm.category_map_id is None:
        algorithm.category_map = category_map
        algorithm.save()
    return algorithm


def setup_algorithms_and_pipeline(
    project: Project,
    detector_name: str,
    classifier_name: str,
    category_labels: list[tuple[str, str]],
) -> tuple[Algorithm, Algorithm, Pipeline]:
    """Create/reuse the detector + classifier Algorithms and a Pipeline grouping them.

    ``detector_name``/``classifier_name`` come from the JSON ``detector_bot``/``identifier_bot``
    fields. ``category_labels`` are the ``(order-name, "ORDER")`` pairs observed in the data.
    """
    category_map = get_or_create_category_map(category_labels) if category_labels else None
    detector = get_or_create_algorithm(detector_name, AlgorithmTaskType.DETECTION)
    classifier = get_or_create_algorithm(classifier_name, AlgorithmTaskType.CLASSIFICATION, category_map=category_map)

    pipeline_name = f"Mothbox — {detector_name}"
    pipeline, _ = Pipeline.objects.get_or_create(
        name=pipeline_name, defaults={"description": "Imported Mothbox results"}
    )
    pipeline.algorithms.add(detector, classifier)
    pipeline.projects.add(project)
    return detector, classifier, pipeline


# --------------------------------------------------------------------------------------
# Deployment provisioning (from JSON metadata)
# --------------------------------------------------------------------------------------
def provision_deployment(
    metadata: dict,
    data_source: S3StorageSource | None = None,
    data_source_subdir: str | None = None,
    data_source_regex: str = r"_HDR0\.jpg$",
) -> Deployment:
    """Get/create the Project + Deployment (+ Site, Device) from a JSON's top-level metadata.

    Keyed on ``deployment_name``. Sets lat/lon, research site and device, and — when a data
    source is given — the S3 sync settings so ``sync_captures`` can register the raw frames.
    """
    project_name = (metadata.get("project") or "Mothbox").strip()
    deployment_name = (metadata.get("deployment_name") or "").strip()
    if not deployment_name:
        raise ValueError("JSON metadata is missing 'deployment_name'; cannot provision a deployment.")

    project, _ = Project.objects.get_or_create(name=project_name)

    site = None
    site_name = (metadata.get("site") or "").strip()
    if site_name:
        site, _ = Site.objects.get_or_create(name=site_name, project=project)

    device = None
    device_name = (metadata.get("device") or "").strip()
    if device_name:
        device, _ = Device.objects.get_or_create(name=device_name, project=project)

    deployment, created = Deployment.objects.get_or_create(
        name=deployment_name,
        project=project,
        defaults={
            "latitude": _to_float(metadata.get("latitude")),
            "longitude": _to_float(metadata.get("longitude")),
            "research_site": site,
            "device": device,
            "description": _deployment_description(metadata),
        },
    )

    # Backfill S3 sync settings (idempotent) so the deployment can pull its raw frames.
    updated_fields: list[str] = []
    if data_source is not None and deployment.data_source_id != data_source.pk:
        deployment.data_source = data_source
        updated_fields.append("data_source")
    subdir = data_source_subdir if data_source_subdir is not None else deployment_name
    if deployment.data_source_subdir != subdir:
        deployment.data_source_subdir = subdir
        updated_fields.append("data_source_subdir")
    if deployment.data_source_regex != data_source_regex:
        deployment.data_source_regex = data_source_regex
        updated_fields.append("data_source_regex")
    if updated_fields:
        deployment.save(update_fields=updated_fields)

    if created:
        logger.info(f"Provisioned deployment '{deployment_name}' in project '{project_name}'")
    return deployment


def _deployment_description(metadata: dict) -> str:
    """Human-readable description from the non-geo metadata (habitat, attractor, UTC, …)."""
    keys = ["habitat", "attractor", "attractor_location", "ground_height", "UTC", "crew", "notes", "deployment_date"]
    parts = [f"{k}: {metadata[k]}" for k in keys if str(metadata.get(k) or "").strip()]
    return "\n".join(parts)


def _to_float(value: typing.Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Per-image import
# --------------------------------------------------------------------------------------
def _patch_url(data_source: S3StorageSource | None, source_image_key: str, patch_path: str) -> str | None:
    """Full public URL of a patch crop, located under ``<night>/_processed/`` beside the raw."""
    if not patch_path:
        return None
    night_dir = posixpath.dirname(source_image_key)
    patch_key = posixpath.join(night_dir, PROCESSED_SUBDIR, patch_path)
    if data_source is not None and data_source.public_base_url:
        import ami.utils.s3 as s3

        return s3.public_url(data_source.config, patch_key)
    # Fall back to a bare key; resolvable only if it lands in the default media storage.
    return patch_key


def import_detections_for_image(
    data: dict,
    source_image: SourceImage,
    detector: Algorithm,
    classifier: Algorithm,
    taxon_cache: dict[tuple[str, str], Taxon],
    summary: ImportSummary,
) -> list[Detection]:
    """Create Detections (+ Classifications) for one ``_botdetection.json`` / SourceImage.

    Idempotent: if the image already has detections from ``detector``, it is skipped. All
    detections/classifications for the image are bulk-created; occurrences are created later.
    """
    if Detection.objects.filter(source_image=source_image, detection_algorithm=detector).exists():
        summary.source_images_skipped_already_imported += 1
        return []

    data_source = source_image.deployment.data_source if source_image.deployment_id else None
    shapes = data.get("shapes") or []

    detections: list[Detection] = []
    # Parallel list of the taxon (or None) each detection should be classified as.
    detection_taxa: list[Taxon | None] = []
    for shape in shapes:
        points = shape.get("points")
        if not points or len(points) < 3:
            continue
        detection = Detection(
            source_image=source_image,
            bbox=bbox_from_points(points),
            timestamp=source_image.timestamp,
            detection_algorithm=detector,
            detection_score=shape.get("confidence_detection"),
            path=_patch_url(data_source, source_image.path, shape.get("patch_path") or ""),
        )
        detections.append(detection)
        # Unidentified "creature" shapes carry no taxonomy → detection only, no classification.
        taxon = None if (shape.get("label") == UNIDENTIFIED_LABEL) else resolve_taxon(shape, taxon_cache)
        detection_taxa.append(taxon)

    if not detections:
        return []

    Detection.objects.bulk_create(detections)
    summary.detections_created += len(detections)

    classifications: list[Classification] = []
    for detection, shape, taxon in zip(detections, shapes, detection_taxa):
        if taxon is None:
            continue
        classifications.append(
            Classification(
                detection=detection,
                taxon=taxon,
                algorithm=classifier,
                category_map=classifier.category_map,
                score=shape.get("confidence_ID"),
                timestamp=source_image.timestamp,
                terminal=True,
            )
        )
    if classifications:
        Classification.objects.bulk_create(classifications)
        summary.classifications_created += len(classifications)

    return detections


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def import_records(
    project: Project,
    detector: Algorithm,
    classifier: Algorithm,
    json_records: typing.Iterable[dict],
    summary: ImportSummary | None = None,
) -> ImportSummary:
    """Import an iterable of parsed ``_botdetection.json`` dicts into an already-synced project.

    Each record must be matched to an existing ``SourceImage`` (by raw-image filename within
    the record's deployment) — images whose raw frame was not registered are logged and
    skipped (the "raw present only" guard). Occurrences + determinations are created at the end
    via Antenna's own bulk helper.
    """
    from ami.ml.models.pipeline import create_and_update_occurrences_for_detections

    summary = summary or ImportSummary()
    taxon_cache: dict[tuple[str, str], Taxon] = {}
    all_detections: list[Detection] = []

    for data in json_records:
        summary.json_files_read += 1
        source_image = _match_source_image(project, data)
        if source_image is None:
            summary.source_images_skipped_no_capture += 1
            continue
        summary.source_images_matched += 1
        detections = import_detections_for_image(data, source_image, detector, classifier, taxon_cache, summary)
        all_detections.extend(detections)

    if all_detections:
        create_and_update_occurrences_for_detections(all_detections, logger=logger)
        summary.occurrences_created += len(all_detections)

    logger.info(f"Mothbox import summary: {summary.as_dict()}")
    return summary


def _match_source_image(project: Project, data: dict) -> SourceImage | None:
    """Find the registered SourceImage for a JSON, by raw-image basename within the project."""
    raw_path = data.get("imagePath") or data.get("filepath") or ""
    basename = posixpath.basename(raw_path.replace("\\", "/"))
    if not basename:
        return None
    # SourceImage.path is the S3 key; match on the trailing filename within this project.
    return SourceImage.objects.filter(project=project, path__endswith=basename).order_by("pk").first()


# --------------------------------------------------------------------------------------
# S3 orchestration — shared by the management command and the MothboxImportJob
# --------------------------------------------------------------------------------------
JSON_REGEX = r"_botdetection\.json$"


def import_from_s3(
    source: S3StorageSource,
    prefix: str = "",
    regex: str = r"_HDR0\.jpg$",
    skip_sync: bool = False,
    logger: logging.Logger = logger,
) -> ImportSummary:
    """End-to-end import of all ``*_botdetection.json`` under ``prefix`` in an S3 source.

    Provisions the Project/Deployment from JSON metadata, registers the raw frames via each
    deployment's ``sync_captures`` (unless ``skip_sync``), sets up the detector/classifier
    Algorithms + Pipeline, and creates Detections/Classifications/Occurrences. Idempotent.
    """
    import ami.utils.s3 as s3

    config = source.config
    # list_files_paginated yields (object_dict | None, count) tuples and defaults to filtering
    # by image extensions, so pass file_extensions to allow .json and skip the trailing sentinel.
    json_keys = [
        obj["Key"]
        for obj, _count in s3.list_files_paginated(
            config, subdir=prefix or None, regex_filter=JSON_REGEX, file_extensions=[".json"]
        )
        if obj is not None
    ]
    logger.info(f"Found {len(json_keys)} detection JSON files under prefix '{prefix}'.")

    records: list[tuple[str, dict]] = []
    for key in json_keys:
        try:
            records.append((key, json.loads(s3.read_file(config, key).decode("utf-8"))))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(f"Skipping unreadable JSON {key}: {exc}")

    summary = ImportSummary()
    if not records:
        return summary

    # Group by deployment, provision + sync each.
    by_deployment: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for key, data in records:
        by_deployment[(data.get("deployment_name") or "").strip()].append((key, data))

    projects: dict[int, Project] = {}
    for deployment_name, dep_records in by_deployment.items():
        if not deployment_name:
            logger.warning(f"Skipping {len(dep_records)} records with no deployment_name.")
            continue
        first_key, first = dep_records[0]
        subdir = _deployment_subdir(config, first_key)
        deployment = provision_deployment(
            first, data_source=source, data_source_subdir=subdir, data_source_regex=regex
        )
        projects[deployment.project_id] = deployment.project
        if not skip_sync:
            logger.info(f"Syncing captures for '{deployment_name}' (subdir='{subdir}') …")
            deployment.sync_captures()

    # Per project: set up algorithms, then import.
    for project in projects.values():
        proj_records = [d for _k, d in records if (d.get("project") or "Mothbox").strip() == project.name]
        detector_name, classifier_name, category_labels = _scan_algorithm_info(proj_records)
        detector, classifier, _pipeline = setup_algorithms_and_pipeline(
            project, detector_name, classifier_name, category_labels
        )
        import_records(project, detector, classifier, proj_records, summary=summary)

    return summary


def _scan_algorithm_info(records: list[dict]) -> tuple[str, str, list[tuple[str, str]]]:
    """Derive detector/classifier names and the ``(label, rank)`` category set from the shapes."""
    detectors: collections.Counter = collections.Counter()
    classifiers: collections.Counter = collections.Counter()
    labels: dict[str, str] = {}
    for data in records:
        for shape in data.get("shapes") or []:
            if shape.get("detector_bot"):
                detectors[shape["detector_bot"]] += 1
            if shape.get("identifier_bot"):
                classifiers[shape["identifier_bot"]] += 1
            if shape.get("label") == UNIDENTIFIED_LABEL:
                continue
            for field, rank in reversed(DWC_RANK_FIELDS):  # deepest populated rank
                name = (shape.get(field) or "").strip()
                if name:
                    labels[name] = rank.value
                    break
    detector_name = detectors.most_common(1)[0][0] if detectors else "Mothbox YOLO detector"
    classifier_name = classifiers.most_common(1)[0][0] if classifiers else "Mothbox classifier"
    return detector_name, classifier_name, sorted(labels.items())


def _deployment_subdir(config, json_key: str) -> str:
    """Bucket subdir (relative to the source prefix) for the deployment folder of a JSON key.

    JSON keys look like ``<deployment>/<night>/_processed/<file>_botdetection.json``; the
    deployment folder is three levels up. The result is made relative to ``config.prefix`` so
    ``sync_captures`` (which prepends the prefix) points at the right place.
    """
    deployment_prefix = posixpath.dirname(posixpath.dirname(posixpath.dirname(json_key)))
    source_prefix = (config.prefix or "").strip("/")
    rel = deployment_prefix
    if source_prefix and deployment_prefix.startswith(source_prefix):
        rel = deployment_prefix[len(source_prefix) :]
    return rel.strip("/")
