"""
Overlay Visual Genome JSON annotations onto images.

Usage:
    python view_images.py <images_folder> <json_folder> [--output <output_folder>]
                         [--no-relationships] [--show]

Bounding boxes and object labels are always drawn.
Relationship arrows are drawn by default (disable with --no-relationships).
Results are saved to <output_folder> (default: ./annotated).
Pass --show to display each image interactively instead of (or as well as) saving.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── colour palette cycling through objects ──────────────────────────────────
PALETTE = [
    (255, 80,  80),   # red
    (80,  160, 255),  # blue
    (80,  220, 80),   # green
    (255, 180, 0),    # orange
    (200, 80,  255),  # purple
    (0,   210, 210),  # cyan
    (255, 255, 80),   # yellow
    (255, 100, 180),  # pink
]


def load_font(size: int = 13):
    """Try to load a truetype font; fall back to the PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def midpoint(box):
    """Return the centre (x, y) of a bounding-box dict with keys x, y, w, h."""
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, colour, width=2):
    """Draw a line with a small arrowhead at *end*."""
    draw.line([start, end], fill=colour, width=width)

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    # Unit vector
    ux, uy = dx / length, dy / length
    # Arrowhead size
    hs = 10
    lx, ly = end[0] - hs * ux, end[1] - hs * uy
    perp_x, perp_y = -uy * hs * 0.4, ux * hs * 0.4
    p1 = (lx + perp_x, ly + perp_y)
    p2 = (lx - perp_x, ly - perp_y)
    draw.polygon([end, p1, p2], fill=colour)


def annotate_image(img: Image.Image, data: dict, draw_rels: bool) -> Image.Image:
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_label = load_font(13)
    font_rel = load_font(11)

    objects = data.get("objects", [])
    # Map object_id → (box_dict, colour)
    id_to_box = {}
    for idx, obj in enumerate(objects):
        colour = PALETTE[idx % len(PALETTE)]
        id_to_box[obj["object_id"]] = (obj, colour)

    # ── Draw bounding boxes ──────────────────────────────────────────────────
    for obj, colour in id_to_box.values():
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        box_colour = colour + (200,)   # semi-transparent fill
        outline_colour = colour + (255,)

        # Filled rect with transparency
        draw.rectangle([x, y, x + w, y + h], outline=outline_colour, width=2)

        label_parts = obj.get("names", [])
        label = label_parts[0] if label_parts else str(obj["object_id"])
        attrs = obj.get("attributes", [])
        if attrs:
            label += f" [{', '.join(attrs[:2])}]"

        # Background pill for text
        try:
            bbox = draw.textbbox((0, 0), label, font=font_label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font_label)

        tx, ty = x + 2, max(0, y - th - 4)
        draw.rectangle([tx - 1, ty - 1, tx + tw + 3, ty + th + 3],
                       fill=(0, 0, 0, 160))
        draw.text((tx, ty), label, font=font_label, fill=colour + (255,))

    # ── Draw relationships ───────────────────────────────────────────────────
    if draw_rels:
        for rel in data.get("relationships", []):
            sub_id = rel.get("subject_id")
            obj_id = rel.get("object_id")
            predicate = rel.get("predicate", "?")
            if sub_id not in id_to_box or obj_id not in id_to_box:
                continue

            sub_box, sub_col = id_to_box[sub_id]
            obj_box, obj_col = id_to_box[obj_id]
            start = midpoint(sub_box)
            end = midpoint(obj_box)

            arrow_col = (220, 220, 60, 200)
            draw_arrow(draw, start, end, arrow_col, width=2)

            # Label at midpoint
            mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
            try:
                rb = draw.textbbox((0, 0), predicate, font=font_rel)
                rw, rh = rb[2] - rb[0], rb[3] - rb[1]
            except AttributeError:
                rw, rh = draw.textsize(predicate, font=font_rel)
            draw.rectangle([mx - 1, my - 1, mx + rw + 3, my + rh + 3],
                           fill=(0, 0, 0, 160))
            draw.text((mx, my), predicate, font=font_rel, fill=(255, 255, 100, 255))

    result = Image.alpha_composite(img, overlay)
    return result.convert("RGB")


def find_image(images_dir: Path, stem: str):
    """Find an image file whose stem matches *stem* (case-insensitive)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Case-insensitive fallback
        candidate = images_dir / f"{stem}{ext.upper()}"
        if candidate.exists():
            return candidate
    return None


def process_pair(image_path: Path, json_path: Path,
                 output_dir: Path | None, draw_rels: bool, show: bool):
    with open(json_path) as f:
        data = json.load(f)

    img = Image.open(image_path)
    annotated = annotate_image(img, data, draw_rels)

    if show:
        annotated.show(title=image_path.name)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{image_path.stem}_annotated.jpg"
        annotated.save(out_path, quality=92)
        print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Visual Genome JSON annotations onto images."
    )
    parser.add_argument("images_dir", help="Folder containing images (1.jpg, 2.jpg, …)")
    parser.add_argument("json_dir",   help="Folder containing JSON annotation files (1.json, 2.json, …)")
    parser.add_argument("--output", "-o", default="annotated",
                        help="Output folder for annotated images (default: annotated)")
    parser.add_argument("--no-relationships", action="store_true",
                        help="Do not draw relationship arrows")
    parser.add_argument("--show", action="store_true",
                        help="Display each annotated image in a viewer")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save output images (useful with --show)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    json_dir   = Path(args.json_dir)

    if not images_dir.is_dir():
        sys.exit(f"Error: images folder not found: {images_dir}")
    if not json_dir.is_dir():
        sys.exit(f"Error: JSON folder not found: {json_dir}")

    output_dir = None if args.no_save else Path(args.output)
    draw_rels  = not args.no_relationships

    json_files = sorted(json_dir.glob("*.json"), key=lambda p: p.stem)
    if not json_files:
        sys.exit(f"No .json files found in {json_dir}")

    print(f"Found {len(json_files)} JSON file(s). Processing…")
    matched = 0
    for json_path in json_files:
        img_path = find_image(images_dir, json_path.stem)
        if img_path is None:
            print(f"  [skip] No image found for {json_path.name}")
            continue
        print(f"  Processing {img_path.name} + {json_path.name}")
        process_pair(img_path, json_path, output_dir, draw_rels, args.show)
        matched += 1

    print(f"\nDone. Processed {matched}/{len(json_files)} pairs.")


if __name__ == "__main__":
    main()
