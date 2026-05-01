
from argparse import ArgumentParser
from pipeline import Standardiser, SceneGraphFusion, FusionConfig, save_scene_graph_json,load_scene_graph_json
from pipeline.wordnet import install_wordnet
from tqdm import tqdm
import os
from pathlib import Path


def get_files_in_folder(folder: Path, depth: int = 2) -> list[Path]:
    """Recursively get all files in a folder up to a certain depth."""
    result = []
    for root, dirs, files in os.walk(folder):
        current_depth = len(Path(root).relative_to(folder).parts)
        if current_depth > depth:
            continue
        for file in files:
            result.append((Path(root).relative_to(folder)) / file)
    return result



def main(args):
    
    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)
    if not folder_a.is_dir():
        print(f"folder '{folder_a.absolute()}' not found")
        return
    if not folder_b.is_dir():
        print(f"folder '{folder_b.absolute()}' not found")
        return
    folder_a_files = sorted(get_files_in_folder(folder_a, depth=args.depth))
    folder_b_files = sorted(get_files_in_folder(folder_b, depth=args.depth))
    print(f"found {len(folder_a_files)} files in folder {folder_a.absolute()} and {len(folder_b_files)} files in folder {folder_b.absolute()}")
    
    folder_a_files_set = set(str(f) for f in folder_a_files)
    folder_b_files_set = set(str(f) for f in folder_b_files)
    common_files = folder_a_files_set.intersection(folder_b_files_set)
    print(f"found {len(common_files)} common files between the two folders")
    
    if not args.output:
        print("no output folder specified, skipping fusion")
        return
        
    fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=args.iou_threshold))
    std = Standardiser(wup_threshold=args.wup_threshold)
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    for filename in tqdm(common_files, desc="Fusing scene graphs"):
        sg_a = load_scene_graph_json(folder_a / filename, source=str(folder_a / filename))
        sg_b = load_scene_graph_json(folder_b / filename, source=str(folder_b / filename))

        # 2. Standardise language across both graphs
        
        std.standardise(sg_a)
        std.standardise(sg_b)

        # 3. Fuse into a single graph
        
        merged = fusion.fuse([sg_a, sg_b])

        # 4. Export the result
        save_scene_graph_json(merged, Path(args.output) / filename)
        if args.visualise:
            merged.visualise()
    


if __name__ == "__main__":
    
    install_wordnet()
    
    parser = ArgumentParser(description="Run the scene graph fusion pipeline on example data.")
    parser.add_argument("folder_a", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/react++",
                        help="Path to JSON file with detections from model A.")
    parser.add_argument("folder_b", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/reltr",
                        help="Path to JSON file with detections from model B.")
    parser.add_argument("-d","--depth", type=int, default=2, help="folder depth to traverse for scene graph JSON files (default: 2)")
    parser.add_argument("-o","--output",help="folder to save the merged graphs to",type=str,default = "merged_graphs")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for bounding box matching during fusion (default: 0.3)")
    parser.add_argument("--wup_threshold", type=float, default=0.85, help="Wu-Palmer similarity threshold for label merging during standardisation (default: 0.85)")
    parser.add_argument("--visualise", action="store_true", help="Visualise the scene graphs before and after fusion")
    args = parser.parse_args()
    main(args)