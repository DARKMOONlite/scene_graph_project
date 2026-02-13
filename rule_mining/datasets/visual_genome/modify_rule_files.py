from typing import Pattern
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re
from dataclasses import dataclass
from rule_mining.datasets.visual_genome.Rule import *

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Replace object IDs with names in triplet files")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input rules file")
    parser.add_argument("-j", "--json", type=str, required=True,help="JSON file or folder containing object ID to name mapping")
    parser.add_argument("-o", "--output", type=str, required=True,help="Output file for named triplets")
    parser.add_argument("-n","--num",type=int,default=-1,help="limit the number of files loaded,-1 means no limit")

    args = parser.parse_args()
    
    try:
        input_path = Path(args.input)
        json_path = Path(args.json)
        output_path = Path(args.output)
        
        data:list[dict] = load_json_files(path=json_path,file_limit=args.num)

        object_id_dict:dict[int,str] = {}
        # Flatten list of dicts into a single dict

        for json_data in tqdm(data, desc="Flattening JSON objects"):
            for obj in json_data.get("objects", []):
                object_id_dict[obj["object_id"]] = obj["names"][0]
        print(f"found {len(object_id_dict)} object instances")
        rules: list[FOLRule] = load_anyBURL_results(input_path)
        modified_rules:list[FOLRule] = replace_object_ids_with_name_in_file(rules=rules, object_id_dict=object_id_dict)
        modified_rules: list[FOLRule] = remove_low_confidence_rules(modified_rules,confidence_threshold=0.5)
        modified_rules:list[FOLRule] = remove_low_occurance_rules(rules=modified_rules,min_successful_occurances=4)
        save_rules_to_file(modified_rules, output_path)
    except Exception as e:
        print(f"Exception: {e}")
        
if __name__ == "__main__":
    main()
