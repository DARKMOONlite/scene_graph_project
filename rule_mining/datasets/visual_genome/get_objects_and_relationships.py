"""
Extract all unique object names and relationship predicates from a folder of
Visual Genome-style JSON annotation files.

Usage:
    python get_objects_and_relationships.py <json_folder> [options]

Options:
    --objects-out   FILE   Write object list to FILE (default: objects.txt)
    --rels-out      FILE   Write relationship list to FILE (default: relationships.txt)
    --no-save              Print to stdout only, do not write files
    --lowercase            Normalise all strings to lowercase
    --counts               Sort by frequency (descending) and show counts
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def process_folder(json_dir: Path, lowercase: bool):
    object_counter: Counter = Counter()
    rel_counter: Counter = Counter()

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        sys.exit(f"No .json files found in {json_dir}")

    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] Could not read {path.name}: {e}", file=sys.stderr)
            continue

        # Objects — collect every name listed for each object entry
        for obj in data.get("objects", []):
            for name in obj.get("names", []):
                name = name.strip()
                if name:
                    object_counter[name.lower() if lowercase else name] += 1

        # Relationships — collect each predicate
        for rel in data.get("relationships", []):
            predicate = rel.get("predicate", "").strip()
            if predicate:
                rel_counter[predicate.lower() if lowercase else predicate] += 1

    return object_counter, rel_counter


def format_entries(counter: Counter, min_counts: int) -> list[str]:
    output = []
    for item,count in counter.most_common():
        if(count<min_counts):
            return output
        output.append(f"{item}\t{count}")




def write_or_print(lines: list[str], label: str, out_path: Path | None):
    if out_path:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {label}: {len(lines)} entries → {out_path}")
    else:
        print(f"\n── {label} ({len(lines)}) ──")
        for line in lines:
            print(line)


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique object names and relationship predicates from VG JSON files."
    )
    parser.add_argument("json_dir", help="Folder containing JSON annotation files")
    parser.add_argument("--objects-out", default="objects.txt", metavar="FILE",
                        help="Output file for object names (default: objects.txt)")
    parser.add_argument("--rels-out", default="relationships.txt", metavar="FILE",
                        help="Output file for relationship predicates (default: relationships.txt)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print results to stdout instead of saving files")
    parser.add_argument("--lowercase", action="store_true",
                        help="Normalise all strings to lowercase before deduplication")
    parser.add_argument("-m","--min",default=1,type=int,help="only include values if they occur >= that this value")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        sys.exit(f"Error: not a directory: {json_dir}")

    print(f"Scanning {json_dir} …")
    object_counter, rel_counter = process_folder(json_dir, args.lowercase)

    obj_lines = format_entries(object_counter, args.min)
    rel_lines = format_entries(rel_counter, args.min)

    obj_out = None if args.no_save else Path(args.objects_out)
    rel_out = None if args.no_save else Path(args.rels_out)

    write_or_print(obj_lines, "Objects", obj_out)
    write_or_print(rel_lines, "Relationships", rel_out)

    print(f"\nSummary: {len(obj_lines)} unique objects, "
          f"{len(rel_lines)} unique relationship types "
          f"(across {sum(rel_counter.values())} total relationship instances)")


if __name__ == "__main__":
    main()
