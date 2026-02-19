"""
python script for segmenting objects from the Visual Genome images based on objects in the accompanying .json file
"""
import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json
from pathlib import Path
import sys
from visual_genome.local import get_scene_graph,Graph,Object
from argparse import ArgumentParser
import pprint
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

sam2_checkpoint = "sam2.1_hiera_small.pt"
sam2_config = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_checkpoint_path = os.path.join(os.path.dirname(sam2.__file__), "..", "checkpoints", sam2_checkpoint)

folder:Path = Path("data/VG_100K/")

files = [entry.name for entry in folder.iterdir() if entry.is_file()]
sorted_files = sorted(files)
print(f"checkpoint path: {sam2_checkpoint_path}")
print(f"config name: {sam2_config}")
predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint_path))


masks = []
for file in tqdm(sorted_files):
    img = Image.open(folder/file)
    graph:Graph = get_scene_graph(int(file.split(".")[0]),"data/raw/","data/by-id/","data/raw/synsets.json")
    
    # attempt to find the objects within the scene using sam2
    # convert VG bounding boxes (x, y, w, h) to SAM2 box prompts (x1, y1, x2, y2)
    boxes = np.array([[obj.x, obj.y, obj.x + obj.width, obj.y + obj.height] for obj in graph.objects])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
    
    # Apply masks to the image as coloured overlays
    img_array = np.array(img.convert("RGB"))
    overlay = img_array.copy()

    rng = np.random.default_rng(42)
    for mask in masks:
        m = mask[0].astype(bool)  # shape (H, W)
        colour = rng.integers(50, 256, size=3)
        overlay[m] = (overlay[m] * 0.4 + colour * 0.6).astype(np.uint8)

    result = Image.fromarray(overlay)
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    break;


print(graph.objects)




