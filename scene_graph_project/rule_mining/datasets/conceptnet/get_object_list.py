"""
Reads a ConceptNet CSV file and collects all unique concept names found in the
subject (col 2) and object (col 3) columns.

Example input line:
  /a/[/r/IsA/,/c/en/0/,/c/en/set_containing_one_element/]  /r/IsA  /c/en/0  /c/en/set_containing_one_element  {...}

Example output (with -l en):
  0
  set_containing_one_element
"""
import sys
from argparse import ArgumentParser, Namespace
from itertools import islice
from tqdm import tqdm


def extract_concept(uri: str, languages: set[str] | None) -> str | None:
    """
    Extract the concept name from a ConceptNet URI like /c/en/some_concept or
    /c/en/some_concept/n/wn/artifact.

    Returns None if the URI language does not match the requested languages
    (or if the URI can't be parsed as a concept).
    """
    parts = uri.strip().split("/")
    # Expected: ['', 'c', '<lang>', '<concept>', ...]
    if len(parts) < 4 or parts[1] != "c":
        return None
    lang = parts[2]
    if languages and lang not in languages:
        return None
    return parts[3]  # just the concept name, no pos/sense suffix


def main() -> None:
    parser = ArgumentParser(
        description="Extract unique concept names from a ConceptNet assertions CSV."
    )
    parser.add_argument("input_file", help="Path to the ConceptNet CSV file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Write one concept per line to this file (default: print to stdout)",
    )
    parser.add_argument(
        "-l", "--languages",
        nargs="*",
        default=None,
        metavar="LANG",
        help="Only keep concepts whose language code matches (e.g. en fr de). "
             "Omit to keep all languages.",
    )
    parser.add_argument(
        "-r", "--relations",
        nargs="*",
        default=None,
        metavar="REL",
        help="Only process lines whose relation matches (e.g. /r/IsA /r/PartOf). "
             "Omit to process all relations.",
    )
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=50_000,
        help="Lines to read per progress-bar update (default: 50 000)",
    )
    args: Namespace = parser.parse_args()

    languages: set[str] | None = set(args.languages) if args.languages else None
    relations: set[str] | None = set(args.relations) if args.relations else None

    print(f"Input:     {args.input_file}", file=sys.stderr)
    print(f"Languages: {languages or 'all'}", file=sys.stderr)
    print(f"Relations: {relations or 'all'}", file=sys.stderr)
    print(file=sys.stderr)

    concepts: set[str] = set()

    with open(args.input_file) as infile:
        with tqdm(desc="Reading", unit="lines", file=sys.stderr) as pbar:
            while True:
                chunk = list(islice(infile, args.chunk_size))
                if not chunk:
                    break
                for line in chunk:
                    fields = line.split("\t")
                    if len(fields) < 4:
                        continue
                    relation = fields[1].strip()
                    if relations and relation not in relations:
                        continue
                    for uri in (fields[2], fields[3]):
                        name = extract_concept(uri, languages)
                        if name:
                            concepts.add(name)
                pbar.update(len(chunk))

    print(f"\nFound {len(concepts):,} unique concepts.", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as outfile:
            for concept in sorted(concepts):
                outfile.write(concept + "\n")
        print(f"Written to: {args.output}", file=sys.stderr)
    else:
        for concept in sorted(concepts):
            print(concept)


if __name__ == "__main__":
    main()
