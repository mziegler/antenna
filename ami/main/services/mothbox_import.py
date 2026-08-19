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
import io
import json
import logging
import math
import posixpath
import typing

from django.utils.text import slugify

from ami.main.models import (
    Classification,
    Deployment,
    Detection,
    Device,
    Occurrence,
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

# Fallback taxon + dedicated algorithm for the "shim" classifications attached to detections
# that Mothbot_Process left unidentified. Everything a Mothbox catches is an arthropod, so
# Arthropoda (phylum) is a safe, correct determination; the shim carries score 0.0 so it's
# trivially distinguishable from real classifications and stays below any positive score
# threshold. See get_or_create_shim_algorithm / shim classifications below.
DEFAULT_UNIDENTIFIED_TAXON = "Arthropoda"
SHIM_ALGORITHM_KEY = "mothbox-import-unidentified-fallback"
SHIM_ALGORITHM_NAME = "Mothbox import — unidentified fallback"
SHIM_SCORE = 0.0

# Reconstruction of captures whose raw frame was deleted: paste the (de-rotated) patch crops
# back onto a solid-grey full-frame canvas at their oriented-box positions, and store the result
# as a half-resolution WebP. WebP crushes the flat background, so a composite is ~1% of the raw
# (~0.17 MB) yet shows the real insects with correctly-oriented, non-occluded patches. The
# composite is a real image object served via the SourceImage's own ``public_base_url``, so no
# Antenna-core/serving change is needed. See reconstruct_composite_webp / _paste_rotated_patch.
RECONSTRUCT_WEBP_QUALITY = 80
RECONSTRUCT_SCALE = 0.5  # half resolution
RECONSTRUCT_BG = (128, 128, 128)  # solid grey background
RECONSTRUCT_EXT = ".webp"
# Capture set that lets users view only real (raw-present) captures; the importer maintains it.
REAL_CAPTURES_COLLECTION_NAME = "Real captures (raw present)"

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
    reconstructed_created: int = 0
    detections_created: int = 0
    classifications_created: int = 0
    shim_classifications_created: int = 0
    occurrences_created: int = 0
    json_files_read: int = 0

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ReconstructionContext:
    """Everything the importer needs to reconstruct a deleted raw frame from its patch crops.

    ``read_config`` reads the patch bytes (the read S3 source, e.g. manu-mothbox);
    ``write_source`` is the S3StorageSource for the dedicated reconstructed bucket (write key +
    ``public_base_url``); the composite is uploaded there and served via that base URL.
    """

    read_config: typing.Any  # ami.utils.s3.S3Config
    write_source: S3StorageSource
    quality: int = RECONSTRUCT_WEBP_QUALITY
    scale: float = RECONSTRUCT_SCALE
    dry_run: bool = False  # skip building/uploading composites (for --dry-run previews)


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

    Matches an existing deployment by its **Mothbox codename** (the S3 folder, stored in
    ``data_source_subdir``) rather than by ``name`` — so renaming a deployment's display name in
    the Antenna UI does NOT make the importer create a duplicate. Sets lat/lon, research site and
    device, and — when a data source is given — the S3 sync settings for ``sync_captures``.
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

    # Match by the Mothbox codename (data_source_subdir), then fall back to name for deployments
    # created before subdir-matching existed; create only if neither is found.
    subdir = data_source_subdir if data_source_subdir is not None else deployment_name
    deployment = (
        Deployment.objects.filter(project=project, data_source_subdir=subdir).first()
        or Deployment.objects.filter(project=project, name=deployment_name).first()
    )
    created = deployment is None
    if created:
        deployment = Deployment.objects.create(
            name=deployment_name,
            project=project,
            latitude=_to_float(metadata.get("latitude")),
            longitude=_to_float(metadata.get("longitude")),
            research_site=site,
            device=device,
            description=_deployment_description(metadata),
        )

    # Backfill S3 sync settings (idempotent) so the deployment can pull its raw frames.
    updated_fields: list[str] = []
    if data_source is not None and deployment.data_source_id != data_source.pk:
        deployment.data_source = data_source
        updated_fields.append("data_source")
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
    unidentified_taxon: Taxon | None = None,
    shim_algorithm: Algorithm | None = None,
) -> list[Detection]:
    """Create Detections (+ Classifications) for one ``_botdetection.json`` / SourceImage.

    Idempotent: if the image already has detections from ``detector``, it is skipped. All
    detections/classifications for the image are bulk-created; occurrences are created later.

    When ``unidentified_taxon``/``shim_algorithm`` are given, detections Mothbot_Process left
    unclassified get a **shim classification** to that taxon at score ``SHIM_SCORE`` (0.0),
    instead of being left determination-less.
    """
    if Detection.objects.filter(source_image=source_image, detection_algorithm=detector).exists():
        summary.source_images_skipped_already_imported += 1
        return []

    data_source = source_image.deployment.data_source if source_image.deployment_id else None

    # (detection, shape, resolved-taxon-or-None) for each usable shape, kept aligned so the
    # classification pass reads the right shape (shapes with <3 points are dropped).
    kept: list[tuple[Detection, dict, Taxon | None]] = []
    for shape in data.get("shapes") or []:
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
        # Unidentified "creature" shapes carry no taxonomy → no real classification.
        taxon = None if (shape.get("label") == UNIDENTIFIED_LABEL) else resolve_taxon(shape, taxon_cache)
        kept.append((detection, shape, taxon))

    if not kept:
        return []

    detections = [d for d, _shape, _taxon in kept]
    Detection.objects.bulk_create(detections)
    summary.detections_created += len(detections)

    classifications: list[Classification] = []
    for detection, shape, taxon in kept:
        if taxon is not None:
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
            summary.classifications_created += 1
        elif unidentified_taxon is not None and shim_algorithm is not None:
            classifications.append(_build_shim_classification(detection, unidentified_taxon, shim_algorithm))
            summary.shim_classifications_created += 1
    if classifications:
        Classification.objects.bulk_create(classifications)

    return detections


def _build_shim_classification(
    detection: Detection, unidentified_taxon: Taxon, shim_algorithm: Algorithm
) -> Classification:
    """A stand-in Classification (score ``SHIM_SCORE``) for a detection with no real ID."""
    return Classification(
        detection=detection,
        taxon=unidentified_taxon,
        algorithm=shim_algorithm,
        category_map=shim_algorithm.category_map,
        score=SHIM_SCORE,
        timestamp=detection.timestamp,
        terminal=True,
    )


def get_or_create_unidentified_taxon(name: str = DEFAULT_UNIDENTIFIED_TAXON) -> Taxon:
    """The fallback taxon for unidentified detections (defaults to Arthropoda, PHYLUM)."""
    rank = TaxonRank.PHYLUM.value if name == DEFAULT_UNIDENTIFIED_TAXON else TaxonRank.UNKNOWN.value
    taxon, _ = Taxon.objects.get_or_create(name=name, defaults={"rank": rank})
    return taxon


def get_or_create_shim_algorithm() -> Algorithm:
    """A dedicated classification Algorithm marking shim (import-fallback) classifications,
    so they are identifiable and removable, and never confused with real classifier output."""
    algorithm, _ = Algorithm.objects.get_or_create(
        key=SHIM_ALGORITHM_KEY,
        defaults={"name": SHIM_ALGORITHM_NAME, "task_type": AlgorithmTaskType.CLASSIFICATION.value},
    )
    return algorithm


# --------------------------------------------------------------------------------------
# Reconstruction of deleted raw captures
# --------------------------------------------------------------------------------------
def _paste_rotated_patch(canvas, patch, points) -> None:
    """Paste a de-rotated patch back onto ``canvas`` at its oriented-box position.

    A Mothbox patch is stored upright; its width is the box's ``|edge12|`` side and its height
    ``|edge01|`` (verified against the data). We resize to those side lengths, rotate by
    ``180 - edge12_angle`` (matches the source frame — the +180 was confirmed visually), and
    paste through the rotated image's own alpha so the transparent expand-corners never occlude
    neighbouring patches. ``canvas``/``patch`` are PIL Images.
    """
    from PIL import Image

    p = points
    l01 = math.hypot(p[1][0] - p[0][0], p[1][1] - p[0][1])  # -> patch height
    l12 = math.hypot(p[2][0] - p[1][0], p[2][1] - p[1][1])  # -> patch width
    img = patch.resize((max(1, round(l12)), max(1, round(l01))), Image.BILINEAR).convert("RGBA")
    angle = 180.0 - math.degrees(math.atan2(p[2][1] - p[1][1], p[2][0] - p[1][0]))
    img = img.rotate(angle, expand=True, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
    cx = sum(q[0] for q in p) / 4.0
    cy = sum(q[1] for q in p) / 4.0
    canvas.paste(img, (int(round(cx - img.width / 2)), int(round(cy - img.height / 2))), img)


def reconstruct_composite_webp(
    data: dict,
    json_key: str,
    read_patch: typing.Callable[[str], bytes],
    quality: int = RECONSTRUCT_WEBP_QUALITY,
    scale: float = RECONSTRUCT_SCALE,
) -> bytes:
    """Rebuild a deleted raw frame as a WebP: grey full-frame canvas + the rotated patch crops.

    ``read_patch(key) -> bytes`` fetches each patch (injected for testability). Patches live in
    the same ``_processed/`` dir as ``json_key``. The canvas is the original raw dimensions
    (from the JSON), downscaled by ``scale`` at the end — the stored ``SourceImage.width/height``
    stay the originals so detection overlays still line up.
    """
    from PIL import Image

    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"JSON {json_key} has no imageWidth/imageHeight; cannot reconstruct.")

    canvas = Image.new("RGB", (width, height), RECONSTRUCT_BG)
    processed_dir = posixpath.dirname(json_key)
    for shape in data.get("shapes", []):
        points = shape.get("points")
        patch_path = shape.get("patch_path")
        if not patch_path or not points or len(points) < 4:
            continue
        try:
            patch = Image.open(io.BytesIO(read_patch(posixpath.join(processed_dir, patch_path)))).convert("RGB")
        except Exception as exc:
            logger.warning(f"Skipping unreadable patch {patch_path} for {json_key}: {exc}")
            continue
        _paste_rotated_patch(canvas, patch, points)

    if scale and scale != 1.0:
        canvas = canvas.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
    buffer = io.BytesIO()
    canvas.save(buffer, format="WEBP", quality=quality)
    return buffer.getvalue()


def _raw_object_key_for_json(json_key: str, data: dict) -> str:
    """The S3 key the (deleted) raw frame would have: its basename in the night dir above _processed/."""
    night_dir = posixpath.dirname(posixpath.dirname(json_key))
    basename = posixpath.basename((data.get("imagePath") or data.get("filepath") or "").replace("\\", "/"))
    return posixpath.join(night_dir, basename)


def _composite_key_for_raw(raw_key: str) -> str:
    return posixpath.splitext(raw_key)[0] + RECONSTRUCT_EXT


def _create_reconstructed_source_image(
    project: Project,
    deployment: Deployment,
    data: dict,
    json_key: str,
    reconstruct: "ReconstructionContext",
    summary: ImportSummary,
) -> SourceImage:
    """Get/create a placeholder-free reconstructed capture: build+upload the composite (once),
    then a SourceImage pointing at it via the reconstructed bucket's ``public_base_url``."""
    from ami.utils.dates import get_image_timestamp_from_filename

    composite_key = _composite_key_for_raw(_raw_object_key_for_json(json_key, data))
    existing = SourceImage.objects.filter(deployment=deployment, path=composite_key).first()
    if existing is not None:
        return existing  # already imported — composite assumed present (idempotent re-run)

    if not reconstruct.dry_run:
        _upload_composite(data, json_key, reconstruct, composite_key)

    source_image = SourceImage.objects.create(
        deployment=deployment,
        path=composite_key,
        project=project,
        public_base_url=reconstruct.write_source.public_base_url,
        width=_to_int(data.get("imageWidth")),
        height=_to_int(data.get("imageHeight")),
        timestamp=get_image_timestamp_from_filename(composite_key),
    )
    summary.reconstructed_created += 1
    return source_image


def _upload_composite(data: dict, json_key: str, reconstruct: "ReconstructionContext", composite_key: str) -> None:
    """Build the composite (unless already in the bucket) and upload it as image/webp."""
    import ami.utils.s3 as s3

    write_cfg = reconstruct.write_source.config
    if s3.file_exists(write_cfg, composite_key):
        return
    composite = reconstruct_composite_webp(
        data,
        json_key,
        read_patch=lambda key: s3.read_file(reconstruct.read_config, key),
        quality=reconstruct.quality,
        scale=reconstruct.scale,
    )
    # write_file() doesn't set a content type; put directly so the .webp serves as image/webp.
    bucket = s3.get_bucket(write_cfg)
    bucket.Object(s3.key_with_prefix(write_cfg, composite_key)).put(Body=composite, ContentType="image/webp")
    logger.info(f"Reconstructed + uploaded composite {composite_key} ({len(composite) / 1e6:.2f} MB)")


def _to_int(value: typing.Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def import_records(
    project: Project,
    detector: Algorithm,
    classifier: Algorithm,
    json_records: typing.Iterable[dict],
    summary: ImportSummary | None = None,
    unidentified_taxon: Taxon | None = None,
    shim_algorithm: Algorithm | None = None,
    reconstruct: "ReconstructionContext | None" = None,
) -> ImportSummary:
    """Import an iterable of parsed ``_botdetection.json`` dicts into an already-synced project.

    Each record is matched to an existing ``SourceImage`` by raw-image filename. If the raw frame
    wasn't registered: when ``reconstruct`` is given, a composite capture is built from the patches
    (see ``ReconstructionContext``); otherwise the record is skipped (the "raw present only" guard).
    Occurrences + determinations are created at the end via Antenna's own bulk helper.

    If ``unidentified_taxon`` is given, unclassified detections get a shim classification to it
    (score ``SHIM_SCORE``); their occurrences are then determined as that taxon with score 0.0.
    Matched (real, raw-present) captures are added to the project's "Real captures" collection.
    """
    from ami.ml.models.pipeline import create_and_update_occurrences_for_detections

    summary = summary or ImportSummary()
    taxon_cache: dict[tuple[str, str], Taxon] = {}
    all_detections: list[Detection] = []
    real_source_images: list[SourceImage] = []
    deployment_cache: dict[str, Deployment] = {}

    for data in json_records:
        summary.json_files_read += 1
        source_image = _match_source_image(project, data)
        if source_image is None:
            if reconstruct is None:
                summary.source_images_skipped_no_capture += 1
                continue
            deployment = _resolve_deployment(project, data, deployment_cache)
            if deployment is None:
                summary.source_images_skipped_no_capture += 1
                continue
            source_image = _create_reconstructed_source_image(
                project, deployment, data, data["_json_key"], reconstruct, summary
            )
        else:
            summary.source_images_matched += 1
            real_source_images.append(source_image)
        detections = import_detections_for_image(
            data, source_image, detector, classifier, taxon_cache, summary, unidentified_taxon, shim_algorithm
        )
        all_detections.extend(detections)

    if all_detections:
        create_and_update_occurrences_for_detections(all_detections, logger=logger)
        summary.occurrences_created += len(all_detections)

    _fix_shim_determination_scores(project, unidentified_taxon)
    _add_to_real_captures_collection(project, real_source_images)

    logger.info(f"Mothbox import summary: {summary.as_dict()}")
    return summary


def _resolve_deployment(project: Project, data: dict, cache: dict[str, Deployment]) -> Deployment | None:
    name = (data.get("deployment_name") or "").strip()
    if not name:
        return None
    if name not in cache:
        # Match by the Mothbox codename (data_source_subdir) first — like provision_deployment —
        # so a deployment renamed in the Antenna UI is still found (matching by name would miss it
        # and wrongly skip the capture as "no raw"). Fall back to name for older deployments.
        cache[name] = (
            Deployment.objects.filter(project=project, data_source_subdir=name).first()
            or Deployment.objects.filter(project=project, name=name).first()
        )
    return cache[name]


def get_or_create_real_captures_collection(project: Project):
    """The manual capture set that lets users filter to only real (raw-present) captures."""
    from ami.main.models import SourceImageCollection

    collection, _ = SourceImageCollection.objects.get_or_create(
        project=project, name=REAL_CAPTURES_COLLECTION_NAME, defaults={"method": "manual"}
    )
    return collection


def _add_to_real_captures_collection(project: Project, source_images: list[SourceImage]) -> None:
    if source_images:
        get_or_create_real_captures_collection(project).images.add(*source_images)


def backfill_real_captures_collection(
    project: Project, batch_size: int = 5000, logger: logging.Logger = logger
) -> int:
    """Add all existing real (raw-present) captures in ``project`` to the "Real captures" set.

    Reconstructed captures are excluded by their ``.webp`` path (only reconstruction writes those).
    Run once before/after enabling reconstruction; the importer keeps the set current thereafter.
    """
    collection = get_or_create_real_captures_collection(project)
    ids = list(
        SourceImage.objects.filter(project=project)
        .exclude(path__endswith=RECONSTRUCT_EXT)
        .values_list("pk", flat=True)
    )
    for i in range(0, len(ids), batch_size):
        collection.images.add(*ids[i : i + batch_size])
    logger.info(f"Added {len(ids)} real captures to '{collection.name}' for {project}")
    return len(ids)


def _fix_shim_determination_scores(project: Project, unidentified_taxon: Taxon | None) -> None:
    """Force shim occurrences' ``determination_score`` to ``SHIM_SCORE`` (0.0).

    ``update_occurrence_determination`` sets the determination taxon but leaves the score
    unset for a score-0 prediction (its ``if new_score:`` guard treats 0.0 as falsy). A NULL
    score is excluded by the default score-threshold filter even at threshold 0, so we set it
    explicitly to 0.0 — that way lowering the project threshold to 0 surfaces these occurrences.
    """
    if unidentified_taxon is None:
        return
    Occurrence.objects.filter(
        project=project, determination=unidentified_taxon, determination_score__isnull=True
    ).update(determination_score=SHIM_SCORE)


def backfill_unidentified_classifications(
    project: Project,
    taxon_name: str = DEFAULT_UNIDENTIFIED_TAXON,
    batch_size: int = 2000,
    logger: logging.Logger = logger,
) -> dict[str, int]:
    """Attach shim classifications to *already-imported* detections that have no classification.

    For every valid detection in ``project`` with no classification, create a shim
    Classification to ``taxon_name`` at score 0.0, then determine each such (undetermined)
    occurrence as that taxon with score 0.0. Idempotent — detections that already have a
    classification are skipped — so it is safe to re-run and to run alongside new imports.
    """
    taxon = get_or_create_unidentified_taxon(taxon_name)
    shim_algorithm = get_or_create_shim_algorithm()
    taxon.projects.add(project)

    det_rows = list(
        Detection.objects.filter(source_image__project=project, bbox__isnull=False, classifications__isnull=True)
        .order_by("pk")
        .values_list("pk", "timestamp")
    )
    logger.info(f"Backfilling shim classifications for {len(det_rows)} unclassified detections in {project}")

    created = 0
    batch: list[Classification] = []
    for pk, timestamp in det_rows:
        batch.append(
            Classification(
                detection_id=pk,
                taxon=taxon,
                algorithm=shim_algorithm,
                category_map=shim_algorithm.category_map,
                score=SHIM_SCORE,
                timestamp=timestamp,
                terminal=True,
            )
        )
        if len(batch) >= batch_size:
            Classification.objects.bulk_create(batch)
            created += len(batch)
            batch = []
    if batch:
        Classification.objects.bulk_create(batch)
        created += len(batch)

    # Determine the (still-undetermined) occurrences those detections belong to. Each imported
    # detection maps to exactly one occurrence, so setting determination directly is correct and
    # avoids update_occurrence_determination's score-0 quirk.
    occ_ids = list(
        Occurrence.objects.filter(
            project=project, determination__isnull=True, detections__classifications__algorithm=shim_algorithm
        )
        .values_list("pk", flat=True)
        .distinct()
    )
    determined = 0
    for i in range(0, len(occ_ids), batch_size):
        determined += Occurrence.objects.filter(pk__in=occ_ids[i : i + batch_size]).update(
            determination=taxon, determination_score=SHIM_SCORE
        )

    logger.info(f"Backfill done: {created} shim classifications, {determined} occurrences determined as {taxon}")
    return {"shim_classifications_created": created, "occurrences_determined": determined}


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
    unidentified_taxon_name: str | None = DEFAULT_UNIDENTIFIED_TAXON,
    reconstruct: "ReconstructionContext | None" = None,
    logger: logging.Logger = logger,
) -> ImportSummary:
    """End-to-end import of all ``*_botdetection.json`` under ``prefix`` in an S3 source.

    Provisions the Project/Deployment from JSON metadata, registers the raw frames via each
    deployment's ``sync_captures`` (unless ``skip_sync``), sets up the detector/classifier
    Algorithms + Pipeline, and creates Detections/Classifications/Occurrences. Idempotent.

    ``unidentified_taxon_name`` (default ``"Arthropoda"``) gives detections Mothbot_Process left
    unclassified a shim classification to that taxon at score 0.0; pass ``None``/`""` to leave
    them determination-less instead.

    ``reconstruct`` (a ``ReconstructionContext``) enables importing captures whose raw frame was
    deleted: a grey composite is built from the patches and uploaded to the reconstructed bucket.
    """
    import ami.utils.s3 as s3

    unidentified_taxon = get_or_create_unidentified_taxon(unidentified_taxon_name) if unidentified_taxon_name else None
    shim_algorithm = get_or_create_shim_algorithm() if unidentified_taxon else None

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
            data = json.loads(s3.read_file(config, key).decode("utf-8"))
            data["_json_key"] = key  # reconstruction needs the key to locate patches + the raw key
            records.append((key, data))
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
    deployments: dict[int, Deployment] = {}
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
        deployments[deployment.pk] = deployment
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
        if unidentified_taxon is not None:
            unidentified_taxon.projects.add(project)
        import_records(
            project,
            detector,
            classifier,
            proj_records,
            summary=summary,
            unidentified_taxon=unidentified_taxon,
            shim_algorithm=shim_algorithm,
            reconstruct=reconstruct,
        )

    # Reconstructed captures are created directly (not via sync_captures), so regroup events to
    # give them an Event and interleave them by timestamp with the real captures — must run
    # before recompute_calculated_fields (which iterates deployment.events).
    if reconstruct is not None:
        from ami.main.models import group_images_into_events

        for deployment in deployments.values():
            group_images_into_events(deployment)

    # Recompute cached counts once, after all detections/occurrences exist. sync_captures
    # runs before occurrences are created, so the deployment/event count fields would
    # otherwise stay stale until the next sync. Scoped to the touched deployments (not the
    # whole project) so a per-night incremental import doesn't re-sweep every deployment.
    for deployment in deployments.values():
        recompute_calculated_fields(deployment, logger=logger)

    return summary


def recompute_calculated_fields(deployment: Deployment, logger: logging.Logger = logger) -> None:
    """Recompute cached count fields for a deployment after an import.

    Mirrors ``Project.update_related_calculated_fields`` but scoped to one deployment:
    refreshes each event's counts, the deployment's own counts (events/captures/detections/
    occurrences/taxa), and the per-image detection counts. These are the project's
    default-filtered counts (see ``Deployment.update_calculated_fields``), so unidentified or
    low-confidence occurrences are intentionally excluded.
    """
    from ami.main.models import update_detection_counts

    logger.info(f"Recomputing calculated fields for deployment {deployment}")
    for event in deployment.events.all():
        event.update_calculated_fields(save=True)
    deployment.update_calculated_fields(save=True)
    if deployment.project_id:
        update_detection_counts(qs=SourceImage.objects.filter(deployment=deployment), project=deployment.project)


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
