from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import sys

ILP_PROJECT_ROOT = Path("~/Documents/phd/inductive_logic_programming/neurosymbolic_ILP").expanduser()
SCENE_GRAPH_ROOT = Path("/mnt/sda1/Datasets/nuscenes/v1.0-mini/scene_graphs/samples/react++")
if str(ILP_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(ILP_PROJECT_ROOT))

from neurosymbolic_pipeline.database_manager import DatabaseManager
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json, collect_scene_graph_files
from scene_graph_project.scene_graph_fusion.pipeline.temporal_stabaliser import TemporalStabaliser
import numpy as np
from PIL import Image




def main(args):
    # files = collect_scene_graph_files(args.input_folder)
    images_tracks:list[tuple[str]] = []
    temporal_stabaliser = TemporalStabaliser()
    db = DatabaseManager(ILP_PROJECT_ROOT / "db/nuscenes.db")
    image_rows = db.get_rows("images",filters={"image_type":"sample"}) # get only images without a previous image, i.e. the first frame of each scene
    initial_images = [row for row in image_rows if row["prev_sample"] is None or row["prev_sample"] == ""] # filter out images that have a previous image
    image_map = {row["image_token"]: row for row in image_rows} # create a map of image_id to image row for easy lookup
    for row in initial_images: # for each unique scene + camera combo
        current_row = row
        images =[]
        while current_row["next_sample"] is not None and current_row["next_sample"] != "":
            # print(current_row["image_token"])
            images.append(current_row["filename"])
            current_row = image_map[current_row["next_sample"]]
        images_tracks.append(tuple(images))
        
        
    print(f"Found {len(images_tracks)} tracks of images. Example track lengths: {[len(track) for track in images_tracks[:10]]}")
    for track_idx, track in enumerate(images_tracks[:1], start=1): # for the first 5 tracks, print the first 5 image filenames
        print(f"\n=== Track {track_idx}/{min(5, len(images_tracks))}: {len(track[:5])} frames selected ===")
        scene_graphs = []
        for frame_idx, path in enumerate(track[:5]):
            scene_graph = load_scene_graph_json(SCENE_GRAPH_ROOT / Path(*Path(path).with_suffix(".json").parts[1:]))
            scene_graphs.append(scene_graph)
            print(
                f"Loaded frame {frame_idx}: {path} | "
                f"objects={len(scene_graph.objects)}, relationships={len(scene_graph.relationships)}"
            )
        
        before_objects = sum(len(graph.objects) for graph in scene_graphs)
        before_relationships = sum(len(graph.relationships) for graph in scene_graphs)
        
        tracked_graphs = temporal_stabaliser.mot_tracking(scene_graphs, images=[np.array(Image.open(path)) for path in track[:5]]) # loads the images as numpy arrays
        # matching_ids:dict[int,int]={}
        # for graph in tracked_graphs:
        #     for obj in graph.objects:
        #         matching_ids[obj.uid] = matching_ids.get(obj.uid, 0) + 1
        #     # graph.visualise()
        # for uid, count in matching_ids.items():
        #     if count > 1:
        #         print(f"Object UID {uid} appears in {count} frames")
        
        # after_objects = sum(len(graph.objects) for graph in tracked_graphs)
        # after_relationships = sum(len(graph.relationships) for graph in tracked_graphs)

        # print(
        #     f"Track {track_idx} summary | "
        #     f"objects: {before_objects} -> {after_objects}, "
        #     f"relationships: {before_relationships} -> {after_relationships}"
        # )
        # tracking_stats = temporal_stabaliser.last_tracking_stats
        # if tracking_stats:
        #     print("  Tracking section:")
        #     print(
        #         "    objects tracked across multiple frames: "
        #         f"{tracking_stats.get('multi_frame_tracks', 0)}"
        #     )
        #     print(
        #         "    detections belonging to multi-frame tracks: "
        #         f"{tracking_stats.get('detections_in_multi_frame_tracks', 0)}"
        #     )
        #     print(
        #         "    recovered missing objects: "
        #         f"{tracking_stats.get('recovered_missing_objects', 0)}"
        #     )
        # for frame_idx, graph in enumerate(tracked_graphs):
        #     print(
        #         f"  frame {frame_idx}: image_id={graph.image_id}, "
        #         f"objects={len(graph.objects)}, relationships={len(graph.relationships)}"
        #     )
    # for file in files:
    #     load_scene_graph_json(file)
# load scene graphs from input folder


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--input_folder", type=Path, required=True, help="Path to the input folder containing scene graphs")
    args = parser.parse_args()
    main(args)


