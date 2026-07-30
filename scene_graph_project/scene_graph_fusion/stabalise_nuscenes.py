from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

ILP_PROJECT_ROOT = Path("~/Documents/phd/inductive_logic_programming/neurosymbolic_ILP").expanduser()
IMAGES_ROOT = Path("/mnt/sda1/Datasets/nuscenes/v1.0-mini/")
if str(ILP_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(ILP_PROJECT_ROOT))

from neurosymbolic_pipeline.database_manager import DatabaseManager
from nuscenes_dev.util.windowing import build_scene_windows_by_length
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json, scene_graph_to_dict
from scene_graph_project.scene_graph_fusion.pipeline.temporal.temporal_stabaliser import TemporalStabaliser
import numpy as np
from PIL import Image
from scene_graph_project.scene_graph_fusion.filter_scene_graphs import filter_scene_graph,OBJECT_BLACKLIST
from scene_graph_project.scene_graph_fusion.pipeline.standardiser import Standardiser


def _compressed_name(first_path: Path, last_path: Path) -> str:
    first_parts = first_path.stem.split("__")
    last_parts = last_path.stem.split("__")
    if first_parts and last_parts:
        first_ts = first_parts[-1]
        last_ts = last_parts[-1]
        first_parts[-1] = f"{first_ts}-{last_ts}"
        return "__".join(first_parts) + ".json"
    return f"{first_path.stem}-{last_path.stem}.json"


def _build_tracks(image_rows: list[dict]) -> list[list[dict]]:
    image_map = {row["image_token"]: row for row in image_rows}
    starts = [row for row in image_rows if row.get("prev") in (None, "")]
    tracks: list[list[dict]] = []
    for start in starts:
        track: list[dict] = []
        current = start
        seen: set[str] = set()
        while current is not None and current["image_token"] not in seen:
            seen.add(current["image_token"])
            track.append(current)
            next_token = current.get("next")
            if not next_token:
                break
            current = image_map.get(next_token)
        if track:
            tracks.append(track)
    return tracks


def _sample_to_sample_windows(
    track: list[dict],
    sample_window_size: int,
    sample_overlap: int,
) -> list[list[dict]]:
    sample_idxs = [i for i, row in enumerate(track) if str(row.get("image_type", "")).lower() == "sample"]
    if len(sample_idxs) < sample_window_size:
        return []

    step = sample_window_size - sample_overlap
    windows: list[list[dict]] = []
    for start in range(0, len(sample_idxs) - sample_window_size + 1, step):
        end = start + sample_window_size
        start_idx = sample_idxs[start]
        end_idx = sample_idxs[end - 1]
        windows.append(track[start_idx:end_idx + 1])
    return windows


def _build_aligned_timestamp_map(image_rows: list[dict]) -> dict[tuple[str, str], str]:
    aligned_by_key: dict[tuple[str, str], str] = {}
    for row in image_rows:
        timestamp = row.get("timestamp")
        channel = row.get("channel")
        aligned_timestamp = row.get("aligned_timestamp")
        if timestamp is None or channel is None or aligned_timestamp in (None, ""):
            continue
        aligned_by_key[(str(timestamp), str(channel))] = str(aligned_timestamp)
    return aligned_by_key


def _aligned_output_frame_path(row: dict, aligned_by_key: dict[tuple[str, str], str]) -> Path:
    filename = row["filename"]
    timestamp = row.get("timestamp")
    channel = row.get("channel")
    if timestamp is None or channel is None:
        return Path(filename)
    aligned_timestamp = aligned_by_key.get((str(timestamp), str(channel)))
    # ponytail: keep original filename when no aligned row exists.
    if not aligned_timestamp:
        return Path(filename)
    path = Path(filename)
    stem_parts = path.stem.split("__")
    if not stem_parts:
        return path
    stem_parts[-1] = aligned_timestamp
    return path.with_name("__".join(stem_parts) + path.suffix)


def _as_int_timestamp(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_window_overlap(size: int, overlap: int, size_flag: str, overlap_flag: str) -> None:
    if size < 1:
        raise ValueError(f"{size_flag} must be at least 1")
    if overlap < 0:
        raise ValueError(f"{overlap_flag} must be at least 0")
    if overlap >= size:
        raise ValueError(f"{overlap_flag} must be smaller than {size_flag}")


def _normalised_entity_prefix(raw_label: str) -> str:
    label = raw_label.split("_", 1)[0] if "_" in raw_label else raw_label
    label = label.strip().lower()
    return label or "object"


def save_scene_graph(graph, output_path: Path) -> None:
    data = scene_graph_to_dict(graph)
    instance_token = output_path.stem
    for obj in data.get("objects", []):
        label = _normalised_entity_prefix(str(obj.get("label", "") or ""))
        scene_graph_id = obj.get("id", "")
        obj["force_label"] = f"{label}_{scene_graph_id}_{instance_token}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def _resolve_scene_graph_path(args, frame_path: Path) -> Path | None:
    graph_rel_path = frame_path.with_suffix(".json")
    if not frame_path.parts:
        return args.input_folder / graph_rel_path if args.input_folder is not None else None

    source = frame_path.parts[0]
    if source == "samples":
        if args.samples_folder is not None:
            return args.samples_folder / Path(*frame_path.parts[1:]).with_suffix(".json")
        if args.input_folder is not None:
            return args.input_folder / graph_rel_path
        return None
    if source == "sweeps":
        if args.sweeps_folder is not None:
            return args.sweeps_folder / Path(*frame_path.parts[1:]).with_suffix(".json")
        if args.input_folder is not None:
            return args.input_folder / graph_rel_path
        return None

    if args.input_folder is not None:
        return args.input_folder / graph_rel_path
    if args.samples_folder is not None:
        return args.samples_folder / graph_rel_path
    if args.sweeps_folder is not None:
        return args.sweeps_folder / graph_rel_path
    return None


def _relative_parent_for_output(args, graph_path: Path) -> Path:
    if args.samples_folder is not None:
        try:
            return graph_path.parent.relative_to(args.samples_folder)
        except ValueError:
            pass
    if args.input_folder is not None:
        try:
            return graph_path.parent.relative_to(args.input_folder / "samples")
        except ValueError:
            pass
    return graph_path.parent


def _build_action_result_windows(
    db: DatabaseManager,
    image_rows: list[dict],
    window_length: int,
    window_overlap: int,
) -> tuple[list[dict], int]:
    scene_windows = build_scene_windows_by_length(
        db=db,
        window_length=window_length,
        window_overlap=window_overlap,
    )
    window_ranges: list[tuple[str, int, int]] = []
    for scene_name, windows in scene_windows.items():
        for window in windows:
            start_ts = min(segment.start_ts for segment in window.segments)
            end_ts = max(segment.end_ts for segment in window.segments)
            window_ranges.append((scene_name, start_ts, end_ts))

    rows_by_channel: dict[str, list[tuple[int, dict]]] = {}
    for row in image_rows:
        channel = row.get("channel")
        aligned_ts = _as_int_timestamp(row.get("aligned_timestamp"))
        if channel in (None, "") or aligned_ts is None:
            continue
        rows_by_channel.setdefault(str(channel), []).append((aligned_ts, row))

    windows: list[dict] = []
    for channel, ts_rows in rows_by_channel.items():
        ts_rows.sort(key=lambda item: item[0])
        for scene_name, start_ts, end_ts in window_ranges:
            window_rows = [row for aligned_ts, row in ts_rows if start_ts <= aligned_ts <= end_ts]
            if not window_rows:
                continue
            windows.append(
                {
                    "channel": channel,
                    "scene_name": scene_name,
                    "start_aligned_timestamp": start_ts,
                    "end_aligned_timestamp": end_ts,
                    "rows": window_rows,
                }
            )
    return windows, len(window_ranges)


def main(args):
    if args.input_folder is None and args.samples_folder is None and args.sweeps_folder is None:
        raise ValueError("Provide at least one of --input_folder, --samples-folder, or --sweeps-folder")
    if args.input_folder is not None and not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")
    if args.samples_folder is not None and not args.samples_folder.exists():
        raise FileNotFoundError(f"Samples folder does not exist: {args.samples_folder}")
    if args.sweeps_folder is not None and not args.sweeps_folder.exists():
        raise FileNotFoundError(f"Sweeps folder does not exist: {args.sweeps_folder}")

    if args.use_action_result_timestamps:
        _validate_window_overlap(args.window_length, args.window_overlap, "--window-length", "--window-overlap")
    else:
        _validate_window_overlap(args.sample_window_size, args.sample_overlap, "--sample-window-size", "--sample-overlap")

    standardiser = Standardiser(blacklist=OBJECT_BLACKLIST)
    temporal_stabaliser = TemporalStabaliser()
    db = DatabaseManager(ILP_PROJECT_ROOT / "db/nuscenes.db")
    image_rows = db.get_rows("images")
    timestamp_to_aligned_timestamp_map = _build_aligned_timestamp_map(image_rows)
    window_specs: list[dict] = []
    if args.use_action_result_timestamps:
        window_specs, unique_range_count = _build_action_result_windows(
            db=db,
            image_rows=image_rows,
            window_length=args.window_length,
            window_overlap=args.window_overlap,
        )
        print(
            f"Found {len(window_specs)} camera windows from action_results "
            f"({unique_range_count} unique windows)"
        )
    else:
        tracks = _build_tracks(image_rows)
        print(f"Found {len(tracks)} full tracks in images table")
        for track_idx, track_rows in enumerate(tracks, start=1):
            windows = _sample_to_sample_windows(
                track_rows,
                sample_window_size=args.sample_window_size,
                sample_overlap=args.sample_overlap,
            )
            if not windows:
                continue
            window_specs.extend(
                {
                    "track_idx": track_idx,
                    "window_idx": window_idx,
                    "rows": window_rows,
                }
                for window_idx, window_rows in enumerate(windows, start=1)
            )

    total_windows = 0
    for idx, window_spec in enumerate(window_specs, start=1):
        window_rows = window_spec["rows"]
        total_windows += 1
        if args.use_action_result_timestamps:
            print(
                f"\n=== Action window {idx}/{len(window_specs)} "
                f"channel={window_spec['channel']} "
                f"range={window_spec['start_aligned_timestamp']}-{window_spec['end_aligned_timestamp']} ==="
            )
        else:
            print(
                f"\n=== Track {window_spec['track_idx']} window "
                f"{window_spec['window_idx']}: {idx}/{len(window_specs)} ==="
            )
        frame_paths = [Path(row["filename"]) for row in window_rows]
        scene_graphs = []
        scene_graph_paths: list[Path] = []
        image_arrays: list[np.ndarray] = []
        skip_window = False
        for frame_idx, frame_path in enumerate(frame_paths):
            graph_rel_path = frame_path.with_suffix(".json")
            is_sweep_frame = bool(frame_path.parts) and frame_path.parts[0] == "sweeps"
            scene_graph_path = _resolve_scene_graph_path(args, frame_path)
            if scene_graph_path is None:
                if is_sweep_frame:
                    continue
                print(f"Skipping window {idx}: no configured scene-graph folder for {frame_path}")
                skip_window = True
                break
            if not scene_graph_path.exists():
                if is_sweep_frame:
                    continue
                print(f"Skipping window {idx}: missing scene graph {scene_graph_path}")
                skip_window = True
                break
            image_path = IMAGES_ROOT / frame_path
            if not image_path.exists():
                print(f"Skipping window {idx}: missing image {image_path}")
                skip_window = True
                break
            scene_graph_paths.append(scene_graph_path)
            scene_graph = load_scene_graph_json(scene_graph_path, source=str(graph_rel_path))
            scene_graphs.append(filter_scene_graph(scene_graph, standardiser))
            image_arrays.append(np.array(Image.open(image_path)))
        if skip_window or not scene_graphs:
            continue

        temporal_graph = temporal_stabaliser.mot_tracking(scene_graphs, images=image_arrays, visualise=args.visualise)
        compressed_graph = temporal_graph.compress()
        print(
            f"Window {idx}/{len(window_specs)} done: "
            f"frames={len(scene_graphs)}, links={temporal_graph.num_links}, "
            f"compressed_objects={len(compressed_graph.objects)}, "
            f"compressed_relationships={len(compressed_graph.relationships)}"
        )
        if args.save:
            if args.use_action_result_timestamps:
                # Use scene_name from action_results so filenames match the exs/bk folder naming convention.
                channel = window_spec["channel"]
                scene_name = window_spec["scene_name"]
                start_ts = window_spec["start_aligned_timestamp"]
                end_ts = window_spec["end_aligned_timestamp"]
                output_path = Path(args.output_folder) / channel / f"{scene_name}__{channel}__{start_ts}-{end_ts}.json"
            else:
                first_graph_path = scene_graph_paths[0]
                first_output_frame_path = _aligned_output_frame_path(window_rows[0], timestamp_to_aligned_timestamp_map)
                last_output_frame_path = _aligned_output_frame_path(window_rows[-1], timestamp_to_aligned_timestamp_map)
                first_output_graph_path = _resolve_scene_graph_path(args, first_output_frame_path) or first_output_frame_path.with_suffix(".json")
                last_output_graph_path = _resolve_scene_graph_path(args, last_output_frame_path) or last_output_frame_path.with_suffix(".json")
                relative_parent = _relative_parent_for_output(args, first_graph_path)
                output_path = Path(args.output_folder) / relative_parent / _compressed_name(
                    first_output_graph_path, last_output_graph_path
                )
            save_scene_graph(compressed_graph, output_path)
            print(f"Saved compressed graph to: {output_path}")
        if args.visualise:
            compressed_graph.visualise()
    print(f"Processed windows: {total_windows}")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_folder",
        type=Path,
        help="Path to a root folder containing scene graphs under samples/ and sweeps/",
    )
    parser.add_argument(
        "--samples-folder",
        type=Path,
        help="Path to scene graphs for sample frames (e.g. .../samples).",
    )
    parser.add_argument(
        "--sweeps-folder",
        type=Path,
        help="Path to scene graphs for sweep frames (e.g. .../sweeps). Optional.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        required=True,
        help="Folder to write compressed graphs",
    )

    parser.add_argument("-v","--visualise", action="store_true", help="Visualise the tracking results")
    parser.add_argument("--save", action="store_true", help="Save each compressed graph as JSON")
    parser.add_argument(
        "--use-action-result-timestamps",
        action="store_true",
        help="Use unique start/end aligned timestamps from action_results and build per-camera windows (inclusive)",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=1,
        help="Number of adjacent action sections to combine into one window (used with --use-action-result-timestamps).",
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=0,
        help="Overlapping sections between consecutive action windows (used with --use-action-result-timestamps).",
    )
    parser.add_argument(
        "--sample-window-size",
        type=int,
        default=2,
        help="How many sample images to include per tracking window (ignored with --use-action-result-timestamps)",
    )
    parser.add_argument(
        "--sample-overlap",
        type=int,
        default=1,
        help="How many sample images consecutive windows share (ignored with --use-action-result-timestamps)",
    )
    args = parser.parse_args()
    main(args)
