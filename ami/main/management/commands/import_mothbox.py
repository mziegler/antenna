"""
Import pre-computed Mothbox / Mothbot_Process results from an S3 bucket into Antenna.

Reads ``*_botdetection.json`` files under a bucket prefix, provisions the Project/Deployment
from their embedded metadata, registers the raw frames as SourceImages via the deployment's
normal ``sync_captures`` flow, then creates Detections + Classifications + Occurrences from the
JSON shapes (see ``ami.main.services.mothbox_import`` for the mapping logic).

The command is idempotent, so the same invocation serves the one-shot backfill and incremental
per-night re-runs. Example:

    python manage.py import_mothbox \\
        --storage-source "hetzner-mothbox" \\
        --prefix "Dataset_ManuNet_RestorationNewerC_fluidRobin_2026-05-04/2026-05-17"

The S3StorageSource (with bucket + credentials + public_base_url) must be created first via the
Django admin or shell.
"""

import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ami.main.models import S3StorageSource
from ami.main.services import mothbox_import as mb

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Import pre-computed Mothbox/Mothbot_Process detections + classifications from S3."

    def add_arguments(self, parser):
        parser.add_argument(
            "--storage-source",
            required=True,
            help="Name or PK of the S3StorageSource holding the Mothbox data.",
        )
        parser.add_argument(
            "--prefix",
            default="",
            help="Bucket sub-prefix to scan for *_botdetection.json (e.g. a deployment or night folder).",
        )
        parser.add_argument(
            "--regex",
            default=r"_HDR0\.jpg$",
            help="Regex selecting raw frames to register as SourceImages (default: _HDR0.jpg).",
        )
        parser.add_argument(
            "--skip-sync", action="store_true", help="Don't sync SourceImages (assume already synced)."
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Parse and report counts, but roll back all writes."
        )
        parser.add_argument(
            "--unidentified-taxon",
            default=mb.DEFAULT_UNIDENTIFIED_TAXON,
            help=(
                "Fallback taxon for detections Mothbot_Process left unclassified — a shim "
                "classification at score 0.0 is created for each (default: Arthropoda). "
                "Pass an empty string to leave them determination-less."
            ),
        )

    def handle(self, *args, **options):
        source = self._resolve_storage_source(options["storage_source"])

        with transaction.atomic():
            summary = mb.import_from_s3(
                source,
                prefix=options["prefix"],
                regex=options["regex"],
                skip_sync=options["skip_sync"],
                unidentified_taxon_name=options["unidentified_taxon"] or None,
            )
            if options["dry_run"]:
                self.stdout.write(self.style.WARNING("Dry run — rolling back all writes."))
                transaction.set_rollback(True)

        self.stdout.write(self.style.SUCCESS(f"Mothbox import complete: {summary.as_dict()}"))

    @staticmethod
    def _resolve_storage_source(identifier: str) -> S3StorageSource:
        source = None
        if identifier.isdigit():
            source = S3StorageSource.objects.filter(pk=int(identifier)).first()
        if source is None:
            source = S3StorageSource.objects.filter(name=identifier).first()
        if source is None:
            raise CommandError(f"No S3StorageSource found matching '{identifier}'.")
        return source
