"""
python script for segmenting objects from the Visual Genome images based on objects in the accompanying .json file
"""
from typing import Any

import torch
import sam2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from multiset import Multiset
import json
from pathlib import Path
import sys
from visual_genome.local import Relationship, get_scene_graph,Graph,Object
from argparse import ArgumentParser
from pprint import pprint
from PIL import Image as PILImage
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from scene_graph_project.rule_mining.datasets.visual_genome.simple_basic_predicates import get_centroid, BasicPredicate
import networkx as nx
SCORE_THRESHOLD = 0.5


sam2_checkpoint = "sam2.1_hiera_small.pt"
sam2_config = "configs/sam2.1/sam2.1_hiera_s.yaml"
sam2_checkpoint_path = os.path.join(os.path.dirname(str(sam2.__file__)), "..", "checkpoints", sam2_checkpoint)

folder:Path = Path("data/VG_100K/")

files = [entry.name for entry in folder.iterdir() if entry.is_file()]
sorted_files = sorted(files)
print(f"checkpoint path: {sam2_checkpoint_path}")
print(f"config name: {sam2_config}")
predictor = SAM2ImagePredictor(build_sam2(sam2_config, sam2_checkpoint_path))
plt.figure(figsize=(12, 8))
plt.title("Visual Genome Scene Graph")
plt.subplot(2,2,1)      
rng = np.random.default_rng(42)




def sam_get_masks(img:PILImage.Image,graph:Graph)->tuple[list[np.ndarray],list[float],dict[int,Any]]:
    # img = Image.open(folder/file)
    boxes = np.array([[obj.x, obj.y, obj.x + obj.width, obj.y + obj.height] for obj in graph.objects])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(img)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        #dictioary that relates
        mask_to_object = {i: graph.objects[i] for i in range(len(masks))}
        
    return (masks,scores,mask_to_object)

def node_colour_list(G:nx.DiGraph)->list[np.ndarray]:
    
    object_colours = [rng.integers(50, 256, size=3)/255.0 for i in range(G.number_of_nodes())]
    return object_colours

def draw_graph(G:nx.Graph,colours:list[np.ndarray],draw_edges:bool=True)->dict:
    pos = nx.spring_layout(G)
    if draw_edges:
        nx.draw(G, pos, with_labels=True, node_color=[tuple(c) for c in colours], node_size=1000, font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=7)
    else:
        nx.draw_networkx_nodes(G,pos,node_color=[tuple(c) for c in colours],node_size=1000)
        nx.draw_networkx_labels(G,pos,font_size=8)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, "label"), font_size=7,font_color="#010057")
    return pos

def draw_overlapping_graphs(G1:nx.Graph,G1_source_identifier:str,G2:nx.Graph,G2_source_identifier:str)->nx.MultiDiGraph:
    combinedGraph:nx.MultiDiGraph = nx.MultiDiGraph()
    for u, v, data in G1.edges(data=True):
        combinedGraph.add_edge(u, v, **data, source=G1_source_identifier)
    # mark original edges explicitly too for clarity
    for u, v, data in G2.edges(data=True):
        combinedGraph.add_edge(u, v, **data, source=G2_source_identifier)
    pos = draw_graph(G2, object_colours,draw_edges=False)
    original_edges = [(u, v, k) for u, v, k, d in combinedGraph.edges(keys=True, data=True) if d.get("source") == G1_source_identifier]
    basic_edges    = [(u, v, k) for u, v, k, d in combinedGraph.edges(keys=True, data=True) if d.get("source") == G2_source_identifier]
    nx.draw_networkx_edges(combinedGraph, pos, edgelist=original_edges, edge_color="#e05c00", arrows=True)
    nx.draw_networkx_edges(combinedGraph, pos, edgelist=basic_edges,    edge_color="#271779", arrows=True, style="dashed", connectionstyle="arc3,rad=0.2")
    original_labels = {(u, v, k): d.get("label", "") for u, v, k, d in combinedGraph.edges(keys=True, data=True) if d.get("source") == G1_source_identifier}
    basic_labels    = {(u, v, k): d.get("label", "") for u, v, k, d in combinedGraph.edges(keys=True, data=True) if d.get("source") == G2_source_identifier}
    # draw_networkx_edge_labels doesn't handle MultiDiGraph (u,v,k) keys reliably,
    # so place labels manually to ensure correct colour per source
    for (u, v, k), label in original_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2 + k * 0.005
        plt.text(x, y, label, fontsize=6, color="#e05c00", ha="center", va="center")
    for (u, v, k), label in basic_labels.items():
        x = (pos[u][0] + pos[v][0]) / 2
        y = (pos[u][1] + pos[v][1]) / 2 - k * 0.005
        plt.text(x, y, label, fontsize=6, color="#271779", ha="center", va="center")
    return combinedGraph
    
def draw_annotated_image(img:PILImage.Image,masks:list[np.ndarray],mask_colours:list[np.ndarray])->None:
    img_array = np.array(img.convert("RGB"))
    overlay = img_array.copy()
    for index, mask in enumerate(masks):
        if scores[index] < 0.5:
            print(f"object {graph.objects[index]} has low confidence : {scores[index]}")

        #? Print off the image with masks
        m = mask[0].astype(bool)  # shape (H, W)
        colour = (np.array(mask_colours[index]) * 255).astype(np.uint8)
        overlay[m] = (overlay[m] * 0.4 + colour * 0.6).astype(np.uint8)
        x,y = get_centroid(m)
        x_text = x+rng.random()*50
        y_text = y+rng.random()*50
        plt.arrow(x,y,x_text-x,y_text-y)
        plt.text(x_text,y_text,graph.objects[index])
    result = PILImage.fromarray(overlay)
    plt.imshow(result)
    plt.axis("off")
    plt.tight_layout()
    

masks = []
for file in tqdm(sorted_files):
    
    graph:Graph = get_scene_graph(int(file.split(".")[0]),"data/raw/","data/by-id/","data/raw/synsets.json")
    img = PILImage.open(folder/file)
    masks,scores,mask_to_object=sam_get_masks(img,graph=graph)
    # print(mask_to_object)
    print(graph.objects)
    # assign a consistent colour per object instance, reused across all visualisations
    
    originalGraph = nx.DiGraph()
    for obj in graph.objects:
        originalGraph.add_node(obj)
    
    for edge in graph.relationships:
        edge:Relationship
        originalGraph.add_edge(edge.object,edge.subject,label=edge.predicate)
    # matplotlib-compatible normalised colours [0,1] for networkx
    object_colours = node_colour_list(originalGraph)
    
    draw_graph(originalGraph,object_colours)
    

    # Apply masks to the image as coloured overlays
    plt.subplot(2,2,2) 
    draw_annotated_image(img=img,masks=masks,mask_colours=object_colours)
    
    
    
    
    predicate_list = np.empty((len(masks),len(masks)),dtype=object)
    for index, mask in enumerate(masks):
        for index2, mask2 in enumerate(masks[index:], start=index):
            p = BasicPredicate(mask,mask2)

            predicate_list[index,index2] = p.primary_predicates()
            predicate_list[index2,index] = p.reverse().primary_predicates()


    
    basicGraph = nx.DiGraph()
    for i, obj in mask_to_object.items():
        basicGraph.add_node(obj)
    for (i,j), value in np.ndenumerate(predicate_list):
        if i != j and value is not None:
            if value.count("disjoint") == 0:
                basicGraph.add_edge(mask_to_object[i], mask_to_object[j], label=" ".join(value))
    pos = nx.spring_layout(basicGraph)
    plt.subplot(2,2,3)    
    plt.title("Basic Predicate Scene Graph")
    
    draw_graph(basicGraph,object_colours)
    
    plt.subplot(2,2,4)  
    plt.title("Combined Scene Graph")
    draw_overlapping_graphs(basicGraph,"basic",originalGraph,"original")


    

    
    
    plt.show()
    break;






