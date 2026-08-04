
from argparse import ArgumentParser
from scene_graph_project.scene_graph_fusion.pipeline import Standardiser, SceneGraphFusion, FusionConfig, save_scene_graph_json,load_scene_graph_json
from scene_graph_project.scene_graph_fusion.pipeline.wordnet import install_wordnet
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import collect_scene_graph_files
from tqdm import tqdm
import os
from pathlib import Path

def main(args):
    
    folders = args.folders
    if len(folders) < 2:
        print("Please provide at least two folders to fuse scene graphs from.")
        return
    folders = [Path(f) for f in folders]
    for folder in folders:
        if not folder.is_dir():
            print(f"folder '{folder.absolute()}' not found")
            return
        
    files = [set(sorted(collect_scene_graph_files(folder, relative=True, depth=args.depth))) for folder in folders]

    common_files: set[Path] = set.intersection(*files)
    print(f"found {len(common_files)} common files between the two folders")
    
    if not args.output:
        print("no output folder specified, skipping fusion")
        return
        
    fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=args.iou_threshold))
    std = Standardiser(wup_threshold=args.wup_threshold)
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    for filename in tqdm(common_files, desc="Fusing scene graphs"):
        
        scene_graphs = [load_scene_graph_json(folder / filename, source=str(folder / filename)) for folder in folders]
        for sg in scene_graphs:
            std.standardise(sg)
            std.blacklist(sg)

        # 3. Fuse into a single graph
        merged = fusion.fuse(scene_graphs)

        # 4. Export the result
        save_scene_graph_json(merged, Path(args.output) / filename)
        if args.visualise:
            merged.visualise()
    


if __name__ == "__main__":
    
    install_wordnet()
    
    parser = ArgumentParser(description="Run the scene graph fusion pipeline on example data.")
    parser.add_argument("folders",type=str,nargs="+",help="folders to fuse scene graphs from")
    parser.add_argument("-o","--output",help="folder to save the merged graphs to",type=str,default = "merged_graphs")
    parser.add_argument("-d","--depth", type=int, default=2, help="folder depth to traverse for scene graph JSON files (default: 2)")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for bounding box matching during fusion (default: 0.3)")
    parser.add_argument("--wup_threshold", type=float, default=0.85, help="Wu-Palmer similarity threshold for label merging during standardisation (default: 0.85)")
    parser.add_argument("--visualise", action="store_true", help="Visualise the scene graphs before and after fusion")
    args = parser.parse_args()
    main(args)