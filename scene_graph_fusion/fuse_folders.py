
from argparse import ArgumentParser
from pipeline import Standardiser, SceneGraphFusion, FusionConfig, save_scene_graph_json,load_scene_graph_json
from tqdm import tqdm
import os
from pathlib import Path

def main(args):
    
    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    if not folder_a.is_dir():
        print(f"folder '{folder_a.absolute()}' not found")
        return
    if not folder_b.is_dir():
        print(f"folder '{folder_b.absolute()}' not found")
        return
    folder_a_files = sorted([f for f in folder_a.iterdir() if f.is_file()])
    folder_b_files = sorted([f for f in folder_b.iterdir() if f.is_file()])
    print(f"found {len(folder_a_files)} files in folder {folder_a.absolute()} and {len(folder_b_files)} files in folder {folder_b.absolute()}")
    
    folder_a_files_set = set(f.name for f in folder_a_files)
    folder_b_files_set = set(f.name for f in folder_b_files)
    folder_a_files_set.intersection(folder_b_files_set)
    common_files = folder_a_files_set.intersection(folder_b_files_set)
    print(f"found {len(common_files)} common files between the two folders")
    
    if not args.output:
        print("no output folder specified, skipping fusion")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for filename in tqdm(common_files, desc="Fusing scene graphs"):
        sg_a = load_scene_graph_json(folder_a / filename, source=str(folder_a / filename))
        sg_b = load_scene_graph_json(folder_b / filename, source=str(folder_b / filename))

        # 2. Standardise language across both graphs
        std = Standardiser(wup_threshold=0.85)
        std.standardise(sg_a)
        std.standardise(sg_b)

        # 3. Fuse into a single graph
        fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=0.3))
        merged = fusion.fuse([sg_a, sg_b])

        # 4. Export the result
        save_scene_graph_json(merged, Path(args.output) / filename)
    
    
    # sg_a = load_scene_graph_json(args.model_a, source="model_a")
    # sg_b = load_scene_graph_json(args.model_b, source="model_b")

    # # 2. Standardise language across both graphs
    # std = Standardiser(wup_threshold=0.85)
    # std.standardise(sg_a)
    # std.standardise(sg_b)

    # sg_a.visualise()
    # sg_b.visualise()
    # # 3. Fuse into a single graph
    # fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=0.3))
    # merged = fusion.fuse([sg_a, sg_b])

    # # 4. Export the result
    # save_scene_graph_json(merged, "merged_graph.json")
    # print(merged.summary())
    # merged.visualise()



if __name__ == "__main__":
    parser = ArgumentParser(description="Run the scene graph fusion pipeline on example data.")
    parser.add_argument("--folder_a", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/react++",
                        help="Path to JSON file with detections from model A.")
    parser.add_argument("--folder_b", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/reltr",
                        help="Path to JSON file with detections from model B.")
    parser.add_argument("-o","--output",help="folder to save the merged graphs to",type=str,default = "merged_graphs")
    args = parser.parse_args()
    main(args)