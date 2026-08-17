"""
Add all existing real (raw-present) captures in a project to its "Real captures" capture set.

Reconstructed captures (composites for deleted raws) are excluded by their ``.webp`` path. Run
this once so users can filter to only real captures; the importer keeps the set current for new
imports. See ``ami.main.services.mothbox_import.backfill_real_captures_collection``.

    python manage.py backfill_real_captures_collection --project ManuNet
"""

import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ami.main.models import Project
from ami.main.services import mothbox_import as mb

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Add existing real (raw-present) captures to the project's 'Real captures' capture set."

    def add_arguments(self, parser):
        parser.add_argument("--project", required=True, help="Project name or PK.")
        parser.add_argument("--dry-run", action="store_true", help="Report the count, then roll back.")

    def handle(self, *args, **options):
        project = self._resolve_project(options["project"])
        with transaction.atomic():
            count = mb.backfill_real_captures_collection(project)
            if options["dry_run"]:
                self.stdout.write(self.style.WARNING("Dry run — rolling back."))
                transaction.set_rollback(True)
        self.stdout.write(self.style.SUCCESS(f"Added {count} real captures to the set for '{project.name}'."))

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
