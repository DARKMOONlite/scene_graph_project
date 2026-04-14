



from argparse import ArgumentParser
from pipeline import Standardiser, SceneGraphFusion, FusionConfig, save_scene_graph_json,load_scene_graph_json
def main(args):
    sg_a = load_scene_graph_json(args.model_a, source="model_a")
    sg_b = load_scene_graph_json(args.model_b, source="model_b")

    # 2. Standardise language across both graphs
    std = Standardiser(wup_threshold=0.85)
    std.standardise(sg_a)
    std.standardise(sg_b)

    sg_a.visualise()
    sg_b.visualise()
    # 3. Fuse into a single graph
    fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=0.3))
    merged = fusion.fuse([sg_a, sg_b])

    # 4. Export the result
    save_scene_graph_json(merged, "merged_graph.json")
    print(merged.summary())
    merged.visualise()



if __name__ == "__main__":
    parser = ArgumentParser(description="Run the scene graph fusion pipeline on example data.")
    parser.add_argument("model_a", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/react++/COCO_train2014_000000000036.json",
                        help="Path to JSON file with detections from model A.")
    parser.add_argument("model_b", type=str, default="/mnt/sda1/Datasets/hl_dataset/scene_graphs/reltr/COCO_train2014_000000000036.json",
                        help="Path to JSON file with detections from model B.")
    args = parser.parse_args()
    main(args)