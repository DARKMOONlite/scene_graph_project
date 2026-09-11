from __future__ import annotations
from argparse import ArgumentParser
from os import getpid
from pathlib import Path
import json
import sys

ILP_PROJECT_ROOT = Path("~/Documents/phd/inductive_logic_programming/neurosymbolic_ILP").expanduser()
IMAGES_ROOT = Path("/mnt/sda1/Datasets/nuscenes/v1.0-mini/")
if str(ILP_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(ILP_PROJECT_ROOT))

from neurosymbolic_pipeline.database_manager import DBRow, DatabaseManager
from nuscenes_dev.util.windowing import build_scene_windows_by_length
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json, scene_graph_to_dict
from scene_graph_project.scene_graph_fusion.pipeline.temporal.temporal_stabaliser import TemporalStabaliser
import numpy as np
from PIL import Image
from scene_graph_project.scene_graph_fusion.filter_scene_graphs import filter_scene_graph,OBJECT_BLACKLIST
from scene_graph_project.scene_graph_fusion.pipeline.standardiser import Standardiser
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import set_start_method


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


def _annotation_scene_names(db: DatabaseManager) -> set[str]:
    return {
        str(row["scene_name"])
        for row in db.get_rows("human_annotations")
        if row.get("scene_name") not in (None, "")
    }


def _filter_rows_by_scene(image_rows: list[dict], scene_names: set[str]) -> list[dict]:
    if any("scene_name" not in row for row in image_rows):
        raise ValueError("The images table must contain a scene_name column for --only-annotations.")
    return [row for row in image_rows if row["scene_name"] in scene_names]


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


def _aligned_output_frame_path(row: dict) -> Path:
    filename = row["filename"]
    aligned_timestamp = row.get("aligned_timestamp")
    # ponytail: keep original filename when no aligned row exists.
    if not aligned_timestamp:
        return Path(filename)
    path = Path(filename)
    stem_parts = path.stem.split("__")
    if not stem_parts:
        return path
    stem_parts[-1] = aligned_timestamp
    stem_parts = [str(part) for part in stem_parts if part]
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


def _resolve_images_root(args) -> Path:
    if args.images_root is not None:
        return args.images_root
    for folder in (args.samples_folder, args.sweeps_folder, args.input_folder):
        if folder is None:
            continue
        for ancestor in (folder, *folder.parents):
            if ancestor.name == "scene_graphs":
                # ponytail: standard nuScenes layouts keep images beside scene_graphs.
                return ancestor.parent
    return IMAGES_ROOT


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
    scene_names: set[str] | None = None,
) -> tuple[list[dict], int]:
    scene_windows = build_scene_windows_by_length(
        db=db,
        window_length=window_length,
        window_overlap=window_overlap,
    )
    if scene_names is not None:
        scene_windows = {
            scene_name: windows
            for scene_name, windows in scene_windows.items()
            if scene_name in scene_names
        }
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


# Each process creates these once, then reuses them for its assigned batch.
_worker_args = None
_worker_images_root: Path | None = None
_worker_standardiser = None
_worker_stabaliser = None


def _initialise_worker(args, images_root: Path) -> None:
    global _worker_args, _worker_images_root, _worker_standardiser, _worker_stabaliser
    _worker_args = args
    _worker_images_root = images_root
    _worker_standardiser = Standardiser(blacklist=OBJECT_BLACKLIST)
    _worker_stabaliser = TemporalStabaliser()

def _process_window_job(window_job: dict) -> tuple[int, int, float, int]:
    args = _worker_args
    window_rows = window_job["rows"]
    frame_paths = [Path(row["filename"]) for row in window_rows]
    scene_graphs = []
    scene_graph_paths: list[Path] = []
    image_arrays: list[np.ndarray] = []
    for frame_path in frame_paths:
        graph_rel_path = frame_path.with_suffix(".json")
        is_sweep_frame = bool(frame_path.parts) and frame_path.parts[0] == "sweeps"
        scene_graph_path = _resolve_scene_graph_path(args, frame_path)
        if scene_graph_path is None or not scene_graph_path.exists():
            if is_sweep_frame:
                continue
            return 0, 0, 0.0, 0
        image_path = _worker_images_root / frame_path
        if not image_path.exists():
            return 0, 0, 0.0, 0
        scene_graph_paths.append(scene_graph_path)
        scene_graph = load_scene_graph_json(scene_graph_path, source=str(graph_rel_path))
        scene_graphs.append(filter_scene_graph(scene_graph, _worker_standardiser))
        image_arrays.append(np.array(Image.open(image_path)))
    if not scene_graphs:
        return 0, 0, 0.0, 0

    temporal_graph = _worker_stabaliser.mot_tracking(scene_graphs, images=image_arrays, visualise=args.visualise)
    compressed_graph = temporal_graph.compress()
    unlinked_objects = sum(len(graph.objects) for graph in temporal_graph.graphs) - sum(
        len(link.instances) for link in temporal_graph.links
    )

    if args.use_action_result_timestamps:
        output_path = (
            Path(args.output_folder)
            / window_job["channel"]
            / (
                f"{window_job['scene_name']}__{window_job['channel']}__"
                f"{window_job['start_aligned_timestamp']}-{window_job['end_aligned_timestamp']}.json"
            )
        )
    else:
        first_output_path = _aligned_output_frame_path(window_rows[0])
        last_output_path = _aligned_output_frame_path(window_rows[-1])
        output_path = Path(args.output_folder) / _relative_parent_for_output(
            args, scene_graph_paths[0]
        ) / _compressed_name(
            _resolve_scene_graph_path(args, first_output_path) or first_output_path.with_suffix(".json"),
            _resolve_scene_graph_path(args, last_output_path) or last_output_path.with_suffix(".json"),
        )
    save_scene_graph(compressed_graph, output_path)
    print(f"Worker {getpid()} completed a window", flush=True)
    return 1, temporal_graph.num_links, (
        temporal_graph.num_links / unlinked_objects if unlinked_objects else 0.0
    ), int(bool(unlinked_objects))


def _process_window_jobs(window_jobs: list[dict]) -> tuple[int, int, float, int]:
    if not window_jobs:
        return 0, 0, 0.0, 0
    return tuple(sum(values) for values in zip(*(_process_window_job(job) for job in window_jobs)))


def _process_track_batch(tracks: list[list[dict]]) -> tuple[int, int, float, int]:
    # Keep every window from a track in the same process.
    window_jobs = [
        {"rows": window}
        for track in tracks
        for window in _sample_to_sample_windows(
            track, _worker_args.sample_window_size, _worker_args.sample_overlap
        )
    ]
    return _process_window_jobs(window_jobs)


def main(args):
    if args.input_folder is None and args.samples_folder is None and args.sweeps_folder is None:
        raise ValueError("Provide at least one of --input_folder, --samples-folder, or --sweeps-folder")
    if args.input_folder is not None and not args.input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input_folder}")
    if args.samples_folder is not None and not args.samples_folder.exists():
        raise FileNotFoundError(f"Samples folder does not exist: {args.samples_folder}")
    if args.sweeps_folder is not None and not args.sweeps_folder.exists():
        raise FileNotFoundError(f"Sweeps folder does not exist: {args.sweeps_folder}")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    if args.visualise and args.threads > 1:
        raise ValueError("--visualise requires --threads 1")

    if args.use_action_result_timestamps:
        _validate_window_overlap(args.window_length, args.window_overlap, "--window-length", "--window-overlap")
    else:
        _validate_window_overlap(args.sample_window_size, args.sample_overlap, "--sample-window-size", "--sample-overlap")

    db = DatabaseManager(args.db)
    image_rows:list[DBRow] = db.get_rows("images")
    annotated_scenes = _annotation_scene_names(db) if args.only_annotations else None
    if annotated_scenes is not None:
        image_rows = _filter_rows_by_scene(image_rows, annotated_scenes)
        print(f"Restricting processing to {len(annotated_scenes)} scenes in human_annotations")
    images_root = _resolve_images_root(args)
    if args.use_action_result_timestamps:
        window_jobs, unique_range_count = _build_action_result_windows(
            db=db,
            image_rows=image_rows,
            window_length=args.window_length,
            window_overlap=args.window_overlap,
            scene_names=annotated_scenes,
        )
        print(
            f"Found {len(window_jobs)} camera windows from action_results "
            f"({unique_range_count} unique windows)"
        )

        batches = [window_jobs[i:i + args.batch_size] for i in range(0, len(window_jobs), args.batch_size)]
        worker = _process_window_jobs
    else:
        tracks = _build_tracks(image_rows)
        print(f"Found {len(tracks)} full tracks in images table")
        # Split whole tracks as evenly as possible across worker processes.
        batches = [tracks[i:i + args.batch_size] for i in range(0, len(tracks), args.batch_size)]
        worker = _process_track_batch

    if batches:
        set_start_method("spawn", force=True)
        with ProcessPoolExecutor(max_workers=args.threads,initializer=_initialise_worker,initargs=(args, images_root),) as executor:
            futures = [executor.submit(worker, batch) for batch in batches] # type: ignore
            results = [
                future.result()
                # Advance when a batch finishes, rather than when it is submitted.
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches")
            ]
    else:
        results = []

    successful_windows = sum(result[0] for result in results)
    total_links = sum(result[1] for result in results)
    total_links_unlinked_ratio = sum(result[2] for result in results)
    ratio_window_count = sum(result[3] for result in results)
    average_links = total_links / successful_windows if successful_windows else 0
    average_links_unlinked_ratio = total_links_unlinked_ratio / ratio_window_count if ratio_window_count else 0
    print(f"Successful windows: {successful_windows}")
    print(f"Average links found: {average_links}")
    print(f"Average links/unlinked ratio: {average_links_unlinked_ratio}")

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
        "--images-root",
        type=Path,
        help="Path containing the samples/ and sweeps/ image folders. Defaults to the parent of scene_graphs.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=Path,
        required=True,
        help="Folder to write compressed graphs",
    )
    parser.add_argument("--db", type=Path, help="Path to the database file",default=ILP_PROJECT_ROOT / "db/nuscenes.db")
    parser.add_argument(
        "--only-annotations",
        action="store_true",
        help="Only stabilise scenes listed in human_annotations.scene_name.",
    )
    parser.add_argument("-v","--visualise", action="store_true", help="Visualise the tracking results")
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
    parser.add_argument("--threads", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of windows to process per worker (default: 5)")
    args = parser.parse_args()
    main(args)
