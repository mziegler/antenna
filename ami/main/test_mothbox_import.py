"""Tests for the Mothbox / Mothbot_Process importer (ami.main.services.mothbox_import)."""

import datetime
import io

from django.test import TestCase
from PIL import Image

from ami.main.models import (
    Classification,
    Deployment,
    Detection,
    Occurrence,
    S3StorageSource,
    SourceImage,
    SourceImageCollection,
    Taxon,
    TaxonRank,
)
from ami.main.services import mothbox_import as mb
from ami.tests.fixtures.main import setup_test_project

# A rotated ("oriented") box from a real _botdetection.json — its axis-aligned envelope is
# the min/max of the four corner points.
DIPTERA_POINTS = [[5157.2, 4710.3], [6568.9, 2425.7], [4220.4, 974.6], [2808.8, 3259.2]]
DIPTERA_ENVELOPE = [2808.8, 974.6, 6568.9, 4710.3]

RAW_FILENAME = "fluidRobin_2026_05_17__18_04_06_HDR0.jpg"


def _botdetection_json() -> dict:
    """A minimal _botdetection.json: one order-identified shape + one unidentified 'creature'."""
    return {
        "imagePath": f"/home/mb/.../2026-05-17/{RAW_FILENAME}",
        "imageWidth": 9248,
        "imageHeight": 6944,
        "project": "Mothbox Test",
        "site": "restorationNewerC",
        "device": "fluidRobin",
        "deployment_name": "Dataset_ManuNet_RestorationNewerC_fluidRobin_2026-05-04",
        "latitude": "-12.891751",
        "longitude": "-71.406923",
        "shapes": [
            {
                "label": "ORDER_Diptera",
                "shape_type": "rotation",
                "direction": 31.7,
                "points": DIPTERA_POINTS,
                "patch_path": f"{RAW_FILENAME[:-4]}_1_Mothbot.pt.jpg",
                "confidence_detection": 0.588,
                "identifier_bot": "pybioclip_2.1.3",
                "confidence_ID": 0.669,
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "order": "Diptera",
            },
            {
                "label": mb.UNIDENTIFIED_LABEL,
                "shape_type": "rotation",
                "direction": 0.0,
                "points": [[100, 100], [200, 100], [200, 200], [100, 200]],
                "patch_path": f"{RAW_FILENAME[:-4]}_0_Mothbot.pt.jpg",
                "confidence_detection": 0.42,
                "identifier_bot": "",
            },
        ],
    }


class MothboxImportTest(TestCase):
    def setUp(self):
        self.project, self.deployment = setup_test_project(reuse=False)
        self.source_image = SourceImage.objects.create(
            project=self.project,
            deployment=self.deployment,
            timestamp=datetime.datetime(2026, 5, 17, 18, 4, 6),
            path=f"test/2026-05-17/{RAW_FILENAME}",
            width=9248,
            height=6944,
        )
        self.detector, self.classifier, _pipeline = mb.setup_algorithms_and_pipeline(
            self.project,
            detector_name="Mothbot_yolo11m_4500_imgsz1600_b1_2024-01-18.pt",
            classifier_name="pybioclip_2.1.3",
            category_labels=[("Diptera", "ORDER")],
        )

    def test_bbox_from_points_is_axis_aligned_envelope(self):
        self.assertEqual(mb.bbox_from_points(DIPTERA_POINTS), DIPTERA_ENVELOPE)

    def test_import_creates_detections_classifications_occurrences(self):
        summary = mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])

        # Both shapes become detections; only the identified one gets a classification.
        # (setup_test_project seeds unrelated fixture data, so scope counts to our image.)
        our_detections = Detection.objects.filter(source_image=self.source_image)
        our_classifications = Classification.objects.filter(detection__source_image=self.source_image)
        self.assertEqual(summary.detections_created, 2)
        self.assertEqual(summary.classifications_created, 1)
        self.assertEqual(our_detections.count(), 2)
        self.assertEqual(our_classifications.count(), 1)

        # The oriented box is stored as its axis-aligned envelope.
        classified = our_classifications.get()
        self.assertEqual(classified.detection.bbox, DIPTERA_ENVELOPE)
        self.assertAlmostEqual(classified.score, 0.669)
        self.assertEqual(classified.detection.detection_score, 0.588)

        # Taxonomy resolved to the order, with the parent chain built.
        diptera = classified.taxon
        self.assertEqual(diptera.name, "Diptera")
        self.assertEqual(diptera.rank, TaxonRank.ORDER.value)
        self.assertEqual(diptera.parent.name, "Insecta")
        self.assertTrue(Taxon.objects.filter(name="Arthropoda").exists())

        # An occurrence per detection, and the determination is the classified taxon.
        our_occurrences = Occurrence.objects.filter(detections__source_image=self.source_image).distinct()
        self.assertEqual(our_occurrences.count(), 2)
        self.assertEqual(our_occurrences.filter(determination=diptera).count(), 1)

        # The reused Mothbox patch is stored as the detection crop.
        self.assertTrue(classified.detection.path.endswith("_1_Mothbot.pt.jpg"))

    def test_import_is_idempotent(self):
        mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])
        # Re-running must not create duplicates.
        summary = mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])
        self.assertEqual(summary.detections_created, 0)
        self.assertEqual(summary.source_images_skipped_already_imported, 1)
        self.assertEqual(Detection.objects.filter(source_image=self.source_image).count(), 2)

    def test_unmatched_source_image_is_skipped(self):
        data = _botdetection_json()
        data["imagePath"] = "/some/other/NONEXISTENT_frame.jpg"
        summary = mb.import_records(self.project, self.detector, self.classifier, [data])
        self.assertEqual(summary.source_images_skipped_no_capture, 1)
        self.assertEqual(Detection.objects.filter(source_image=self.source_image).count(), 0)

    def test_shim_classification_for_unidentified_detections(self):
        taxon = mb.get_or_create_unidentified_taxon("Arthropoda")
        shim = mb.get_or_create_shim_algorithm()
        summary = mb.import_records(
            self.project,
            self.detector,
            self.classifier,
            [_botdetection_json()],
            unidentified_taxon=taxon,
            shim_algorithm=shim,
        )
        # The Diptera box gets a real classification; the "creature" box gets a shim.
        self.assertEqual(summary.classifications_created, 1)
        self.assertEqual(summary.shim_classifications_created, 1)
        shim_cls = Classification.objects.filter(algorithm=shim, detection__source_image=self.source_image)
        self.assertEqual(shim_cls.count(), 1)
        self.assertEqual(shim_cls.get().taxon, taxon)
        self.assertEqual(shim_cls.get().score, 0.0)
        # The creature occurrence is now determined as Arthropoda at score 0.0 (not NULL).
        occ = Occurrence.objects.filter(determination=taxon, detections__source_image=self.source_image).distinct()
        self.assertEqual(occ.count(), 1)
        self.assertEqual(occ.get().determination_score, 0.0)

    def test_backfill_adds_shims_to_existing_unclassified(self):
        # Import WITHOUT the shim (old behavior) → the creature box is left unclassified.
        mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])
        self.assertTrue(
            Detection.objects.filter(source_image=self.source_image, classifications__isnull=True).exists()
        )
        mb.backfill_unidentified_classifications(self.project, taxon_name="Arthropoda")
        taxon = mb.get_or_create_unidentified_taxon("Arthropoda")
        # No unclassified detection remains on our image, and its occurrence is now determined.
        self.assertFalse(
            Detection.objects.filter(source_image=self.source_image, classifications__isnull=True).exists()
        )
        occ = Occurrence.objects.filter(determination=taxon, detections__source_image=self.source_image).distinct()
        self.assertEqual(occ.count(), 1)
        self.assertEqual(occ.get().determination_score, 0.0)

    def test_recompute_calculated_fields_runs_and_populates_deployment_counts(self):
        # After import + recompute, the deployment's cached count fields are populated ints
        # (recompute delegates to Antenna's update_calculated_fields; this checks the wiring).
        mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])
        mb.recompute_calculated_fields(self.deployment)
        self.deployment.refresh_from_db()
        for field in ("events_count", "captures_count", "detections_count", "occurrences_count", "taxa_count"):
            self.assertIsInstance(getattr(self.deployment, field), int, field)

    def test_provision_deployment_from_metadata(self):
        deployment = mb.provision_deployment(_botdetection_json())
        self.assertEqual(deployment.name, "Dataset_ManuNet_RestorationNewerC_fluidRobin_2026-05-04")
        self.assertEqual(deployment.project.name, "Mothbox Test")
        self.assertAlmostEqual(deployment.latitude, -12.891751)
        self.assertAlmostEqual(deployment.longitude, -71.406923)
        self.assertEqual(deployment.research_site.name, "restorationNewerC")
        self.assertEqual(deployment.device.name, "fluidRobin")
        # Re-provisioning is idempotent (no duplicate deployment).
        again = mb.provision_deployment(_botdetection_json())
        self.assertEqual(again.pk, deployment.pk)

    def test_provision_matches_renamed_deployment_by_codename(self):
        # Import creates the deployment, then a user renames its display name in Antenna.
        deployment = mb.provision_deployment(_botdetection_json())
        codename = deployment.data_source_subdir
        deployment.name = "RestorationNewerC: full brightness"
        deployment.save()
        # A re-import of the same Mothbox data (same codename) must re-link to the renamed
        # deployment by data_source_subdir, not create a duplicate — and keep the friendly name.
        again = mb.provision_deployment(_botdetection_json())
        self.assertEqual(again.pk, deployment.pk)
        self.assertEqual(again.name, "RestorationNewerC: full brightness")
        self.assertEqual(Deployment.objects.filter(project=deployment.project, data_source_subdir=codename).count(), 1)

    def test_reconstruct_missing_raw_creates_composite_capture(self):
        write_source = S3StorageSource.objects.create(
            name="recon",
            bucket="recon-bucket",
            access_key="",
            secret_key="",
            public_base_url="https://obj.example.com/recon/",
        )
        # dry_run=True skips the S3 build/upload but still creates the SourceImage.
        ctx = mb.ReconstructionContext(read_config=None, write_source=write_source, dry_run=True)
        rec = _botdetection_json()
        rec["deployment_name"] = self.deployment.name  # so _resolve_deployment finds our deployment
        rec["imagePath"] = "/mb/2026-05-17/RECON_2026_05_17__19_00_00_HDR0.jpg"  # no matching raw SourceImage
        rec[
            "_json_key"
        ] = "manu-net-deployments/D/2026-05-17/_processed/RECON_2026_05_17__19_00_00_HDR0_botdetection.json"

        summary = mb.import_records(self.project, self.detector, self.classifier, [rec], reconstruct=ctx)

        self.assertEqual(summary.reconstructed_created, 1)
        self.assertEqual(summary.source_images_matched, 0)
        si = SourceImage.objects.get(deployment=self.deployment, path__endswith=".webp")
        self.assertTrue(si.path.endswith("RECON_2026_05_17__19_00_00_HDR0.webp"))
        self.assertEqual(si.public_base_url, "https://obj.example.com/recon/")
        self.assertEqual((si.width, si.height), (9248, 6944))  # original raw dims from the JSON
        self.assertEqual(si.timestamp, datetime.datetime(2026, 5, 17, 19, 0, 0))
        self.assertEqual(Detection.objects.filter(source_image=si).count(), 2)

        # Idempotent: re-running doesn't create a second composite capture.
        summary2 = mb.import_records(self.project, self.detector, self.classifier, [rec], reconstruct=ctx)
        self.assertEqual(summary2.reconstructed_created, 0)
        self.assertEqual(SourceImage.objects.filter(deployment=self.deployment, path__endswith=".webp").count(), 1)

    def test_reconstruct_resolves_renamed_deployment_by_codename(self):
        # Reproduces the field bug: a deployment renamed in the Antenna UI keeps its Mothbox
        # codename in data_source_subdir, but the JSON still carries the codename as
        # deployment_name. _resolve_deployment must match by codename, or missing-raw captures
        # are wrongly skipped as "no capture" instead of reconstructed.
        codename = "Dataset_ManuNet_RestorationNewerC_fluidRobin_2026-05-04"
        self.deployment.data_source_subdir = codename
        self.deployment.name = "RestorationNewerC: full brightness"
        self.deployment.save()

        write_source = S3StorageSource.objects.create(
            name="recon2",
            bucket="recon-bucket",
            access_key="",
            secret_key="",
            public_base_url="https://obj.example.com/recon/",
        )
        ctx = mb.ReconstructionContext(read_config=None, write_source=write_source, dry_run=True)
        rec = _botdetection_json()
        rec["deployment_name"] = codename  # the source data keeps the ugly codename
        rec["imagePath"] = "/mb/2026-05-17/RECON_2026_05_17__19_00_00_HDR0.jpg"  # no matching raw
        rec[
            "_json_key"
        ] = "manu-net-deployments/D/2026-05-17/_processed/RECON_2026_05_17__19_00_00_HDR0_botdetection.json"

        summary = mb.import_records(self.project, self.detector, self.classifier, [rec], reconstruct=ctx)

        self.assertEqual(summary.reconstructed_created, 1)
        self.assertEqual(summary.source_images_skipped_no_capture, 0)
        self.assertEqual(SourceImage.objects.filter(deployment=self.deployment, path__endswith=".webp").count(), 1)

    def test_real_captures_collection_gets_matched_captures(self):
        mb.import_records(self.project, self.detector, self.classifier, [_botdetection_json()])
        coll = SourceImageCollection.objects.get(project=self.project, name=mb.REAL_CAPTURES_COLLECTION_NAME)
        self.assertEqual(coll.method, "manual")
        self.assertIn(self.source_image, list(coll.images.all()))


class MothboxHelpersTest(TestCase):
    def test_reconstruct_composite_webp_produces_halfres_webp(self):
        buf = io.BytesIO()
        Image.new("RGB", (60, 40), (200, 30, 30)).save(buf, "JPEG")
        patch_bytes = buf.getvalue()
        data = {
            "imageWidth": 400,
            "imageHeight": 300,
            "shapes": [{"points": [[100, 100], [160, 100], [160, 140], [100, 140]], "patch_path": "p0.jpg"}],
        }
        out = mb.reconstruct_composite_webp(
            data, "night/_processed/x.json", read_patch=lambda k: patch_bytes, scale=0.5
        )
        img = Image.open(io.BytesIO(out))
        self.assertEqual(img.format, "WEBP")
        self.assertEqual(img.size, (200, 150))  # half of 400x300; overlays use the stored originals

    def test_deployment_subdir_is_relative_to_source_prefix(self):
        class _Config:
            prefix = "mothbox/"

        key = "mothbox/Dataset_X/2026-05-17/_processed/frame_botdetection.json"
        self.assertEqual(mb._deployment_subdir(_Config(), key), "Dataset_X")

    def test_scan_algorithm_info_picks_names_and_labels(self):
        records = [_botdetection_json()]
        detector, classifier, labels = mb._scan_algorithm_info(records)
        self.assertEqual(detector, "Mothbox YOLO detector")  # no detector_bot in fixture shapes
        self.assertEqual(classifier, "pybioclip_2.1.3")
        self.assertEqual(labels, [("Diptera", "ORDER")])
