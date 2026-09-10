"""Overlay a canonical scene graph's bounding boxes and relationships on an image.

Usage:
    python scene_graph_project/scene_graph_fusion/overlay_scene_graph.py GRAPH OUTPUT [--images IMAGE_ROOT]
"""

from __future__ import annotations

import math
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scene_graph_project.scene_graph_fusion.pipeline import SceneGraph, load_scene_graph_json

PALETTE = (
    (255, 80, 80),
    (80, 160, 255),
    (80, 220, 80),
    (255, 180, 0),
    (200, 80, 255),
    (0, 210, 210),
    (255, 255, 80),
    (255, 100, 180),
)
EDGE_COLOUR = (255, 255, 100)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]) -> None:
    draw.line((start, end), fill=EDGE_COLOUR, width=2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if not length:
        return
    unit_x, unit_y = dx / length, dy / length
    base_x, base_y = end[0] - 10 * unit_x, end[1] - 10 * unit_y
    perpendicular_x, perpendicular_y = -4 * unit_y, 4 * unit_x
    draw.polygon(
        (end, (base_x + perpendicular_x, base_y + perpendicular_y),
         (base_x - perpendicular_x, base_y - perpendicular_y)),
        fill=EDGE_COLOUR,
    )


def overlay(graph: SceneGraph, image: Image.Image, font_size: int = 14) -> Image.Image:
    """Return *image* annotated with every object that has a bounding box."""
    result = image.convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    objects = [(obj, PALETTE[index % len(PALETTE)])
               for index, obj in enumerate(graph.objects) if obj.bbox]
    by_uid = {obj.uid: obj for obj, _ in objects}

    for relationship in graph.relationships:
        subject = by_uid.get(relationship.subject_uid)
        object_ = by_uid.get(relationship.object_uid)
        if not subject or not object_:
            continue
        start, end = subject.bbox.centre, object_.bbox.centre
        _arrow(draw, start, end)
        label = relationship.canonical_predicate or relationship.predicate
        width, height = _text_size(draw, label, font)
        x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        draw.rectangle((x - 1, y - 1, x + width + 2, y + height + 2), fill="black")
        draw.text((x, y), label, fill=EDGE_COLOUR, font=font)

    for object_, colour in objects:
        box = object_.bbox
        draw.rectangle((box.x_min, box.y_min, box.x_max, box.y_max), outline=colour, width=2)
        label = object_.canonical_label or object_.label
        width, height = _text_size(draw, label, font)
        x, y = box.x_min, max(0, box.y_min - height - 3)
        draw.rectangle((x - 1, y - 1, x + width + 2, y + height + 2), fill="black")
        draw.text((x, y), label, fill=colour, font=font)

    return result


def image_for_graph(graph_path: Path, image_root: Path) -> Path:
    """Find the image with the same filename stem as *graph_path*."""
    # ponytail: one recursive scan per graph; add an index only if large batches need it.
    matches = sorted(
        path for path in image_root.rglob(f"{graph_path.stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError(f"no image named '{graph_path.stem}' below {image_root}")
        raise ValueError(f"multiple images named '{graph_path.stem}' below {image_root}: {matches}")
    return matches[0]


def main() -> None:
    parser = ArgumentParser(description="Overlay a canonical scene graph JSON file on its source image.")
    parser.add_argument("graph", type=Path, help="Scene graph JSON file.")
    parser.add_argument("output", type=Path, help="Annotated output image.")
    parser.add_argument(
        "--images",
        type=Path,
        help="Root folder containing source images (default: the graph's folder).",
    )
    parser.add_argument("--font-size", type=int, default=14)
    args = parser.parse_args()

    if not args.graph.is_file():
        parser.error(f"scene graph not found: {args.graph}")
    image_root = args.images or args.graph.parent
    if not image_root.is_dir():
        parser.error(f"image folder not found: {image_root}")
    try:
        image_path = image_for_graph(args.graph, image_root)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    annotated = overlay(
        load_scene_graph_json(args.graph), Image.open(image_path), args.font_size
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(args.output)


if __name__ == "__main__":
    main()
