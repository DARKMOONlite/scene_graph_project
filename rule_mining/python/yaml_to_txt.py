import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Generator, Iterable, List
import yaml
description = """Convert Visual Genome objects JSON/YAML to triple-format TXT."""


def _iter_json_array(file_path: Path) -> Generator[Dict, None, None]:
    decoder = json.JSONDecoder()
    buffer = ""
    with file_path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith("["):
                buffer = buffer[1:]
                break

        while True:
            if not buffer:
                chunk = handle.read(65536)
                if not chunk:
                    break
                buffer += chunk
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                break
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            try:
                item, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = handle.read(65536)
                if not chunk:
                    raise
                buffer += chunk
                continue
            yield item
            buffer = buffer[index:]


def load_items(file_path: Path) -> Iterable[dict]:
    if file_path.suffix.lower() in {".yaml", ".yml"}:
        with file_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(stream=handle)
        if not isinstance(data, list):
            raise ValueError("Expected a top-level list in YAML file.")
        return data

    return _iter_json_array(file_path)


def _slug(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )


def _emit_triples(items: Iterable[Dict]) -> Generator[str, None, None]:
    for entry in items:
        image_id = entry.get("image_id")
        if image_id is None:
            continue
        image_subject = f"/vg/image/{image_id}"
        objects: List[Dict] = entry.get("objects", [])
        for obj in objects:
            object_id = obj.get("object_id")
            if object_id is None:
                continue
            object_subject = f"/vg/object/{object_id}"
            yield f"{image_subject}\t/vg/has_object\t{object_subject}"

            for name in obj.get("names", []) or []:
                if not name:
                    continue
                yield f"{object_subject}\t/vg/object/name\t/vg/name/{_slug(name)}"

            for synset in obj.get("synsets", []) or []:
                if not synset:
                    continue
                yield f"{object_subject}\t/vg/object/synset\t/vg/synset/{_slug(synset)}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="vg_to_triples", description=description)
    parser.add_argument("filename", help="Objects JSON/YAML file to load")
    parser.add_argument("-o", "--output", help="Output TXT file", default="triples.txt")
    parser.add_argument("-n", "--num", help="Maximum number of images to process", type=int)

    args: argparse.Namespace = parser.parse_args()

    file_path = Path(args.filename)
    if not file_path.absolute().resolve().exists():
        print(f"file '{file_path.absolute()}' not found")
        sys.exit(1)

    destination_path = Path(args.output)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_items(file_path)
    count = 0
    with destination_path.open("w", encoding="utf-8") as handle:
        for triple in _emit_triples(items):
            handle.write(triple + "\n")
            if args.num is not None:
                count += 1
                if count >= args.num:
                    break


if __name__ == "__main__":
    main()