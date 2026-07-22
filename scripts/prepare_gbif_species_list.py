#!/usr/bin/env python3
"""
Convert a GBIF DarwinCore species-list export into a CSV that Antenna's ``import_taxa``
management command can ingest.

Mothbox study sites ship a GBIF export (tab- or comma-separated) with columns like
``taxonKey, scientificName, taxonRank, kingdom, phylum, class, order, family, genus,
species, speciesKey, ...``. ``import_taxa`` reads per-rank name columns (lowercase) plus an
optional ``gbif_taxon_key``. This script keeps the rank columns and renames ``taxonKey`` →
``gbif_taxon_key``, dropping everything else.

Usage:
    python scripts/prepare_gbif_species_list.py INPUT.csv OUTPUT.csv

Then, inside the Django container:
    python manage.py import_taxa OUTPUT.csv --format csv --list "Manu GBIF"
"""

import argparse
import csv
import sys

RANK_COLUMNS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
OUTPUT_COLUMNS = RANK_COLUMNS + ["gbif_taxon_key", "author"]


def sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;").delimiter
    except csv.Error:
        return "\t" if sample.count("\t") >= sample.count(",") else ","


def convert(input_path: str, output_path: str) -> int:
    with open(input_path, newline="", encoding="utf-8-sig") as fin:
        delimiter = sniff_delimiter(fin.read(8192))
        fin.seek(0)
        reader = csv.DictReader(fin, delimiter=delimiter)
        if reader.fieldnames is None:
            raise SystemExit("Input file has no header row.")
        # GBIF headers are already the rank names; map taxonKey → gbif_taxon_key.
        lower = {name.lower(): name for name in reader.fieldnames}
        key_col = lower.get("taxonkey")
        author_col = lower.get("scientificname")

        written = 0
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for row in reader:
                out = {col: (row.get(lower.get(col, ""), "") or "").strip() for col in RANK_COLUMNS}
                if not any(out.values()):
                    continue  # skip rows with no taxonomy at all
                out["gbif_taxon_key"] = (row.get(key_col, "") or "").strip() if key_col else ""
                out["author"] = (row.get(author_col, "") or "").strip() if author_col else ""
                writer.writerow(out)
                written += 1
    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to the GBIF species-list export (CSV or TSV).")
    parser.add_argument("output", help="Path to write the import_taxa-ready CSV.")
    args = parser.parse_args(argv)
    count = convert(args.input, args.output)
    print(f"Wrote {count} taxa rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
