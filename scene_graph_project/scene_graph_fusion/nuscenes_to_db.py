from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, TypedDict
from functools import lru_cache

from pipeline.database_manager import DatabaseManager


try:
	from nuscenes.nuscenes import NuScenes
except ImportError as exc:  # pragma: no cover - import-time guard
	raise ImportError(
		"nuscenes-devkit is required. Install it with 'pip install nuscenes-devkit'."
	) from exc

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation as R
from tqdm import tqdm
from pipeline.util.util import (
    Sample,CameraSample, CAMERA_CHANNELS, anno2d_to_scene_object, organize_bounding_boxes_by_camera, project_annotations_to_2d,
)

from pipeline import SceneGraph, load_scene_graph_json, ObjectMatch, SceneObject, SceneGraphFusion, FusionConfig




def _normalise_scene_name(name: str) -> str:
    return name.replace("_", "-")


@lru_cache(maxsize=8)
def _build_window_graph_index(scene_graphs_dir: Path) -> list[tuple[str, str, int, int, Path]]:
    index: list[tuple[str, str, int, int, Path]] = []
    for path in scene_graphs_dir.rglob("*.json"):
        stem = path.stem
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        scene_name, channel, ts_range = parts
        if "-" not in ts_range:
            continue
        start_s, end_s = ts_range.split("-", 1)
        try:
            start_ts = int(start_s)
            end_ts = int(end_s)
        except ValueError:
            continue
        index.append((_normalise_scene_name(scene_name), channel, start_ts, end_ts, path))
    return index


def get_image_scene_graph(sample: Sample, camera: CameraSample, scene_graphs_dir: Path) -> SceneGraph:
    # Backward-compatible path: one scene graph per image under samples/... layout.
    file_name = list(Path(camera.filename).with_suffix(".json").parts)
    file_name[0] = scene_graphs_dir
    file_path = Path(*file_name)
    if file_path.exists():
        print(file_path)
        return load_scene_graph_json(file_path)

    # New naming: <scene>__<CAM_*>__<start>-<end>.json, pick window containing frame timestamp.
    try:
        frame_ts = int(Path(camera.filename).stem.split("__")[-1])
    except ValueError as exc:
        raise FileNotFoundError(f"Could not parse timestamp from camera filename: {camera.filename}") from exc

    scene_key = _normalise_scene_name(sample.scene_name)
    channel = CAMERA_CHANNELS.filter(camera.filename)
    for candidate_scene, candidate_channel, start_ts, end_ts, candidate_path in _build_window_graph_index(scene_graphs_dir):
        if (
            candidate_scene == scene_key
            and candidate_channel == channel
            and start_ts <= frame_ts <= end_ts
        ):
            print(candidate_path)
            return load_scene_graph_json(candidate_path)

    raise FileNotFoundError(
        f"No scene graph found for scene={sample.scene_name}, channel={channel}, timestamp={frame_ts}"
    )




def match_annotations_to_scene_graph(sample:Sample) -> dict[str,list[ObjectMatch]]:
    
        result:dict[str,list[ObjectMatch]] = {channel: [] for channel in CAMERA_CHANNELS}
        anno = project_annotations_to_2d(sample, (1600, 900)) # compute 2D bounding boxes for each annotation in the sample for each camera
        anno_per_camera = organize_bounding_boxes_by_camera(anno, (1600, 900)) # group the annotations by camera channel, so we have a list of 2D bounding boxes for each camera
        
        
        
        for camera, anno_list in anno_per_camera.items(): # for each camera in the scene
            scene_objects:list[SceneObject] = [anno2d_to_scene_object(anno) for anno in anno_list]

            
            temp_cam_graph = SceneGraph(image_id=sample.scene_name,objects=scene_objects,relationships=[]) # create a temporary scene graph with only the 2D bounding boxes as objects and no relationships
            # #? get the scene graph from the image and fuse it with the 2D bounding boxes we just computed
            try:
                scene_graph = get_image_scene_graph(sample, sample.cameras[camera], Path(args.dataroot) / args.scene_graph) # get the scene graph for the image from the scene graph directory
            except FileNotFoundError:
                print(f"No scene graph found for camera {camera} in sample {sample.scene_name}. Skipping.")
                continue
            fusion = SceneGraphFusion(FusionConfig(semantic_threshold=0.0,iou_threshold=0.6))
            matches:list[ObjectMatch] = fusion.match([scene_graph,temp_cam_graph])
            print(f"Camera: {camera}, Matches: {len(matches)}")
            for match in matches:
                print(f"Connecting {match.obj_a} with {match.obj_b} in scene graph for image {sample.cameras[camera].filename}")
            result[camera] = matches
        return result


def write_matches_to_db(
    db: DatabaseManager,
    sample: Sample,
    matches_per_camera: dict[str, list[ObjectMatch]],
) -> None:
    """Write scene-graph match info into a dedicated ``scene_graph_observations`` table.

    Only observations that were matched to a scene-graph object produce a row,
    so the table is a strict subset of ``observations``.  Each row records:

    - ``annotation_token``: FK to the ``observations`` table.
    - ``camera_channel``: which camera the match was found in.
    - ``image``: path to the camera image file that yielded the match.
    - ``scene_graph_label``: label of the matched scene-graph object.
    - ``scene_graph_id``: original ID of the matched scene-graph object.

    Args:
        db: Open :class:`DatabaseManager` instance.
        sample: The NuScenes sample whose annotations were matched.
        matches_per_camera: Output of ``match_annotations_to_scene_graph`` for
            this sample — maps camera channel → list of ``ObjectMatch``.
    """
    instance_token_by_annotation: dict[str, str] = {
        ann.token: ann.instance_token for ann in sample.annotations
    }

    rows = [
        {
            "annotation_token": str(match.obj_b.uid),
            "instance_token": instance_token_by_annotation.get(str(match.obj_b.uid), ""),
            "camera_channel": camera,
            "image": sample.cameras[camera].filename,
            "scene_graph_label": match.obj_a.label,
            "scene_graph_id": str(match.obj_a.original_identifier),
        }
        for camera, matches in matches_per_camera.items()
        for match in matches
    ]
    if not rows:
        return
    db.create_table_and_upsert_rows(
        table_name="scene_graph_observations",
        columns={
            "annotation_token": "TEXT",
            "instance_token": "TEXT",
            "camera_channel": "TEXT",
            "image": "TEXT",
            "scene_graph_label": "TEXT",
            "scene_graph_id": "TEXT",
        },
        rows=rows,
        primary_key=("annotation_token", "camera_channel"),
        indexes=[("instance_token",), ("scene_graph_id",), ("image",)],
        append_columns=True
    )


def main(args) -> None:
    nusc = NuScenes(
		version=args.version,
		dataroot=args.dataroot,
		verbose=not args.quiet,
	)
    db = DatabaseManager(args.db_path)

    samples: list[Sample] = [
        Sample.from_nuscenes_sample(nusc, sample["token"])
        for sample in nusc.sample
    ]

    
    
    for sample in tqdm(samples[:]):
        results = match_annotations_to_scene_graph(sample)
        write_matches_to_db(db, sample, results)

    print(f"Scene-graph match data written to {args.db_path}")
    print(f"")
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Connect NuScenes scene graphs to JSON files.")
    parser.add_argument("--dataroot", type=str, default="/mnt/sda1/Datasets/nuscenes/v1.0-mini", help="Path to the NuScenes dataset root directory.")
    parser.add_argument("--scene_graph",type=str,default="scene_graphs/stabalised_graphs_action",help="Path to the scene graph directory from the dataroot.")
    parser.add_argument(
		"--version",
		default="v1.0-mini",
		help="nuScenes dataset version (for example: v1.0-mini, v1.0-trainval).",
	)
    
    parser.add_argument(
        "--db_path",
        default="db/nuscenes.db",
        help="Path to the SQLite database written by src/nuscenes_dev/track.py.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output from NuScenes initialization.")
    args = parser.parse_args()
    main(args)