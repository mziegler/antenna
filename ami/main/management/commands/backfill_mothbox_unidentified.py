"""
Attach shim "unidentified" classifications to already-imported Mothbox detections.

Mothbot_Process leaves many detections unclassified (label ``creature``, no taxonomy). Those
import as detections with no classification, so their occurrences are undetermined — which
Antenna's default filters hide (count 0) and whose detail endpoint 404s. This command gives
every such existing detection a shim classification to a fallback taxon (default Arthropoda)
at score 0.0, and determines its occurrence accordingly. New imports do this automatically
(``import_mothbox --unidentified-taxon``); this backfills data imported before that.

    python manage.py backfill_mothbox_unidentified --project ManuNet
    python manage.py backfill_mothbox_unidentified --project 3 --taxon Arthropoda --dry-run

Idempotent: detections that already have a classification are skipped.
"""

import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ami.main.models import Project
from ami.main.services import mothbox_import as mb

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Attach shim fallback-taxon classifications to Mothbox detections that have none."

    def add_arguments(self, parser):
        parser.add_argument("--project", required=True, help="Project name or PK.")
        parser.add_argument(
            "--taxon",
            default=mb.DEFAULT_UNIDENTIFIED_TAXON,
            help="Fallback taxon name for unclassified detections (default: Arthropoda).",
        )
        parser.add_argument("--dry-run", action="store_true", help="Report counts, then roll back all writes.")

    def handle(self, *args, **options):
        project = self._resolve_project(options["project"])
        with transaction.atomic():
            result = mb.backfill_unidentified_classifications(project, taxon_name=options["taxon"])
            if options["dry_run"]:
                self.stdout.write(self.style.WARNING("Dry run — rolling back all writes."))
                transaction.set_rollback(True)
        self.stdout.write(self.style.SUCCESS(f"Backfill complete for '{project.name}': {result}"))

    @staticmethod
    def _resolve_project(identifier: str) -> Project:
        project = None
        if identifier.isdigit():
            project = Project.objects.filter(pk=int(identifier)).first()
        if project is None:
            project = Project.objects.filter(name=identifier).first()
        if project is None:
            raise CommandError(f"No Project found matching '{identifier}'.")
        return project
