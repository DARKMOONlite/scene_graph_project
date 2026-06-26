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
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json
from scene_graph_project.scene_graph_fusion.pipeline.temporal.temporal_stabaliser import TemporalStabaliser
import numpy as np
from PIL import Image
from scene_graph_project.scene_graph_fusion.filter_scene_graphs import filter_scene_graph,OBJECT_BLACKLIST
from scene_graph_project.scene_graph_fusion.pipeline.standardiser import Standardiser


NUM_TRACKS = 1
NUM_FRAMES_PER_TRACK = 30

def main(args):
    
    standardiser = Standardiser(blacklist=OBJECT_BLACKLIST)
    # files = collect_scene_graph_files(args.input_folder)
    images_tracks:list[tuple[str]] = []
    temporal_stabaliser = TemporalStabaliser()
    db = DatabaseManager(ILP_PROJECT_ROOT / "db/nuscenes.db")
    
    # udate to use non sample images
    
    image_rows = db.get_rows("images") # get only images without a previous image, i.e. the first frame of each scene
    initial_images = [row for row in image_rows if row["prev"] is None or row["prev"] == ""] # filter out images that have a previous image
    image_map = {row["image_token"]: row for row in image_rows} # create a map of image_id to image row for easy lookup
    for row in initial_images: # for each unique scene + camera combo
        current_row = row
        images =[]
        while current_row["next"] is not None and current_row["next"] != "":
            # print(current_row["image_token"])
            images.append(current_row["filename"])
            current_row = image_map[current_row["next"]]
        images_tracks.append(tuple(images))
        
        
    print(f"Found {len(images_tracks)} tracks of images. Example track lengths: {[len(track) for track in images_tracks[:10]]}")
    for track_idx, track in enumerate(images_tracks[:NUM_TRACKS], start=1): # for the first 5 tracks, print the first 5 image filenames
        print(f"\n=== Track {track_idx}/{min(NUM_TRACKS, len(images_tracks))}: {len(track[:NUM_FRAMES_PER_TRACK])} frames selected ===")
        scene_graphs = []
        for frame_idx, path in enumerate(track[:NUM_FRAMES_PER_TRACK]):
            parts = list(Path(path).parts)
            parts.insert(1, SCENE_GRAPH_MODEL)
            path = Path(*parts)
            scene_graph = load_scene_graph_json((SCENE_GRAPH_ROOT/ path).with_suffix(".json"), source=path)
            scene_graphs.append(scene_graph)
            print(
                f"Loaded frame {frame_idx}: {path} | "
                f"objects={len(scene_graph.objects)}, relationships={len(scene_graph.relationships)}"
            )
        
        before_objects = sum(len(graph.objects) for graph in scene_graphs)
        before_relationships = sum(len(graph.relationships) for graph in scene_graphs)
        
        # Filter and standardise the scene graphs
        scene_graphs = [filter_scene_graph(graph, standardiser) for graph in scene_graphs]

        
        temporal_graph = temporal_stabaliser.mot_tracking(scene_graphs, 
            images=[np.array(Image.open(IMAGES_ROOT / path)) for path in track[:NUM_FRAMES_PER_TRACK]],visualise=args.visualise) # loads the images as numpy arrays
        print(f"Tracked graphs: objects={sum(len(graph.objects) for graph in temporal_graph.graphs)}, relationships={sum(len(graph.relationships) for graph in temporal_graph.graphs)}")
        print(f"Links: {temporal_graph.num_links}")
        for link_idx, link in enumerate(temporal_graph.links[:], start=1):
            print(f"Link {link_idx}: {link.class_}: instances={len(link.instances)}, class={link.class_}")
        compressed_graph = temporal_graph.compress()
        print(f"Compressed graph: objects={len(compressed_graph.objects)}, relationships={len(compressed_graph.relationships)}")
        if args.visualise:
            compressed_graph.visualise()

if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--input_folder", type=Path, required=True, help="Path to the input folder containing scene graphs")
    parser.add_argument("-v","--visualise", action="store_true", help="Visualise the tracking results")
    args = parser.parse_args()
    main(args)


