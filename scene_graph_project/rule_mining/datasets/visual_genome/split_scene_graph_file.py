#!/home/sebastian/Documents/scene_graph/scene_graph_project/.venv/bin/python3
""" 
Splits the scene_graphs.json file into a folder of individual scene graphs for better intake
"""
import visual_genome.local as vg


vg.save_scene_graphs_by_id(data_dir="data/raw/",image_data_dir="data/by-id/")

scene_graphs = vg.get_scene_graphs(start_index=0, end_index=-1, min_rels=1,
                                data_dir='data/raw/', image_data_dir='data/by-id/')
print(len(scene_graphs))
print(scene_graphs[0].objects)