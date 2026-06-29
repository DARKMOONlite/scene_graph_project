from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import sys

ILP_PROJECT_ROOT = Path("~/Documents/phd/inductive_logic_programming/neurosymbolic_ILP").expanduser()
SCENE_GRAPH_ROOT = Path("/mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs")
SCENE_GRAPH_MODEL = "merged"
IMAGES_ROOT = Path("/mnt/sda1/Datasets/nuscenes/v1.0-mini/")
if str(ILP_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(ILP_PROJECT_ROOT))

from neurosymbolic_pipeline.database_manager import DatabaseManager
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json, save_scene_graph_json
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


def _with_aligned_timestamp(filename: str, aligned_timestamp: str) -> Path:
    path = Path(filename)
    stem_parts = path.stem.split("__")
    if not stem_parts:
        return path
    stem_parts[-1] = aligned_timestamp
    return path.with_name("__".join(stem_parts) + path.suffix)


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
    return _with_aligned_timestamp(str(filename), aligned_timestamp)


def _as_int_timestamp(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_action_result_windows(image_rows: list[dict], action_rows: list[dict]) -> tuple[list[dict], int]:
    # Aggregate timestamps per (scene_name, section_id) using min(start)/max(end), mirroring
    # get_scene_segment_time_ranges, so output filenames match the exs folder naming convention.
    section_ranges: dict[tuple[str, int], list[int]] = {}  # (scene_name, section_id) -> [min_start, max_end]
    for row in action_rows:
        scene_name = row.get("scene_name", "")
        section_id = row.get("section_id")
        start_ts = _as_int_timestamp(row.get("start_aligned_timestamp"))
        end_ts = _as_int_timestamp(row.get("end_aligned_timestamp"))
        if start_ts is None or end_ts is None or not scene_name or section_id is None:
            continue
        key = (scene_name, int(section_id))
        if key not in section_ranges:
            section_ranges[key] = [start_ts, end_ts]
        else:
            section_ranges[key][0] = min(section_ranges[key][0], start_ts)
            section_ranges[key][1] = max(section_ranges[key][1], end_ts)

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
        for (scene_name, _section_id), (start_ts, end_ts) in sorted(section_ranges.items()):
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
    return windows, len(section_ranges)


def main(args):
    if not args.use_action_result_timestamps:
        if args.sample_window_size < 2:
            raise ValueError("--sample-window-size must be at least 2")
        if args.sample_overlap < 0:
            raise ValueError("--sample-overlap must be at least 0")
        if args.sample_overlap >= args.sample_window_size:
            raise ValueError("--sample-overlap must be smaller than --sample-window-size")

    standardiser = Standardiser(blacklist=OBJECT_BLACKLIST)
    temporal_stabaliser = TemporalStabaliser()
    db = DatabaseManager(ILP_PROJECT_ROOT / "db/nuscenes.db")
    image_rows = db.get_rows("images")
    timestamp_to_aligned_timestamp_map = _build_aligned_timestamp_map(image_rows)
    window_specs: list[dict] = []
    if args.use_action_result_timestamps:
        action_rows = db.get_rows("action_results")
        window_specs, unique_range_count = _build_action_result_windows(image_rows, action_rows)
        print(
            f"Found {len(window_specs)} camera windows from action_results "
            f"({unique_range_count} unique sections)"
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
            for window_idx, window_rows in enumerate(windows, start=1):
                window_specs.append(
                    {
                        "track_idx": track_idx,
                        "window_idx": window_idx,
                        "rows": window_rows,
                    }
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
            parts = list(frame_path.parts)
            parts.insert(1, SCENE_GRAPH_MODEL)
            graph_rel_path = Path(*parts).with_suffix(".json")
            scene_graph_path = SCENE_GRAPH_ROOT / graph_rel_path
            if not scene_graph_path.exists():
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
            if frame_idx == 0 or frame_idx == len(frame_paths) - 1:
                print(
                    f"Window {idx}/{len(window_specs)} frame {frame_idx}: {frame_path} | "
                    f"objects={len(scene_graph.objects)}, relationships={len(scene_graph.relationships)}"
                )
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
                first_output_graph_rel_path = Path(
                    *([*first_output_frame_path.parts[:1], SCENE_GRAPH_MODEL, *first_output_frame_path.parts[1:]])
                ).with_suffix(".json")
                last_output_graph_rel_path = Path(
                    *([*last_output_frame_path.parts[:1], SCENE_GRAPH_MODEL, *last_output_frame_path.parts[1:]])
                ).with_suffix(".json")
                first_output_graph_path = SCENE_GRAPH_ROOT / first_output_graph_rel_path
                last_output_graph_path = SCENE_GRAPH_ROOT / last_output_graph_rel_path
                try:
                    relative_parent = first_graph_path.parent.relative_to(SCENE_GRAPH_ROOT / "samples" / SCENE_GRAPH_MODEL)
                except ValueError:
                    relative_parent = first_graph_path.parent
                output_path = Path(args.output_folder) / relative_parent / _compressed_name(
                    first_output_graph_path, last_output_graph_path
                )
            save_scene_graph_json(compressed_graph, output_path)
            print(f"Saved compressed graph to: {output_path}")
        if args.visualise:
            compressed_graph.visualise()
    print(f"Processed windows: {total_windows}")

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--input_folder", type=Path, required=True, help="Path to the input folder containing scene graphs")
    parser.add_argument("-v","--visualise", action="store_true", help="Visualise the tracking results")
    parser.add_argument("--save", action="store_true", help="Save each compressed graph as JSON")
    parser.add_argument(
        "--use-action-result-timestamps",
        action="store_true",
        help="Use unique start/end aligned timestamps from action_results and build per-camera windows (inclusive)",
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
    parser.add_argument("-o", "--output_folder", type=Path, default=Path("stabalised_graphs"), help="Folder to write compressed graphs")
    args = parser.parse_args()
    main(args)
