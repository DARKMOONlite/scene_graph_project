"""
converts visual genome scene graph files into predicate triplets for each edge in the graph.
"""
import json
import os
import argparse
from pathlib import Path
import pprint
from tqdm import tqdm
from rule_mining.datasets.visual_genome.Rules import *



def object_predicate_to_basic_triplets(predicates:list[dict], objects:list[dict], source:str|None=None)->list[tuple]:
    """strips all non-relevant data out and just stores each relationship as a triplet.
    
    Args:
        predicates: list of relationship dicts from the scene graph
        objects: list of object dicts from the scene graph
        source: optional identifier for the source scene graph file

    Returns:
        List of tuples: (subject_id, predicate, object_id) or
                        (subject_id, predicate, object_id, source) if source is provided
    """
    triples = []
    for pred in predicates:
        subject_id = pred.get("subject_id")
        predicate = pred.get("predicate")
        object_id = pred.get("object_id")
        if subject_id is not None and predicate and object_id is not None:
            if source is not None:
                triples.append((subject_id, predicate, object_id, source))
            else:
                triples.append((subject_id, predicate, object_id))
    return triples

def replace_object_ids_with_name(objects:list[dict],triplets:list[tuple]):
    object_id_dict:dict = {}
    result:list[tuple] = []
    for  obj in objects:
        object_id_dict[obj["object_id"]] = obj["names"]
        # print(f"found obj:{obj['names']}")
    for relationship in triplets:
        subject = object_id_dict.get(relationship[0])
        obj = object_id_dict.get(relationship[2])
        if subject is None or obj is None:
            continue
        if isinstance(subject, list):
            subject = subject[0] if subject else ""
        if isinstance(obj, list):
            obj = obj[0] if obj else ""
        # Preserve any extra fields (e.g. source) beyond the base 3
        extra = relationship[3:]
        result.append((subject, relationship[1], obj) + extra)
        # print(result[-1])
    return result
        
    
    

def save_triplets(triplets_list:list[list[tuple]],file:Path):
    """Save triplets to file, one per line, values separated by spaces."""
    triplets:list[tuple] = [item for sublist in triplets_list for item in sublist] #compress the lists together.
    with open(file, 'w') as f:
        # pprint.pp(object=triplets)
        for triplet in triplets:
            for value in triplet:
                value = str(value)
                f.write(value.replace(" ","_")+'\t')
            f.write("\n")
    print(f"file saved at {file}")
def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-f","--folder","--file",type=str,required=True,help="the folder or specific file that should be loaded")
    parser.add_argument("-n","--num",type=int,default=-1,help="limit the number of files loaded,-1 means no limit")
    parser.add_argument("-o","--output",type=str,help="the file to print the output to")
    
    args:argparse.Namespace = parser.parse_args()
    
    try:
        data = load_json_files(path=Path(args.folder),file_limit=args.num)
        print(f"Successfully loaded {len(data)} JSON file(s)")
        results = []
        for json_data in tqdm(data): #! Not the best method but it works for now
            triples:list[tuple] = object_predicate_to_basic_triplets(predicates=json_data["relationships"],objects=json_data["objects"],source=json_data.get("image_id"))
            triples_with_names = replace_object_ids_with_name(objects=json_data["objects"],triplets=triples)
            results.append(triples_with_names)
        if args.output is not None:
            save_triplets(results,Path(args.output))
        else:
            print("no file saved")
        
    except Exception as e:
        print(f"Error: {e}")
    
    
if __name__ == "__main__":
    main()