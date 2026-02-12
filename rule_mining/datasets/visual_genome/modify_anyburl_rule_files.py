from typing import Pattern
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import re
from dataclasses import dataclass


@dataclass
class AnyBURLRule:
    score1: int
    score2: int
    confidence: float
    head: str
    body: list[str]

    def get_rule_string(self) -> str:
        return f"{self.head} <= {', '.join(self.body)}"

def load_json_files(path:Path,file_limit:int)->list[dict]:
    """Load JSON files from a single file or all JSON files in a folder."""
    
    
    if path.is_file():
        # Single file
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                return [json.load(f)]
        else:
            raise ValueError(f"File {path} is not a JSON file")
    
    elif path.is_dir():
        # Folder - load all JSON files
        json_files = list(path.glob('*.json'))
        if not json_files:
            raise ValueError(f"No JSON files found in {path}")
        
        data:list = []
        limit:int = 0
        if(file_limit >0):
            json_files = json_files[:file_limit]
        for json_file in tqdm(sorted(json_files)):
            # print(f"Loading {json_file.name}...")
            with open(json_file, 'r') as f:
                data.append(json.load(f))
        return data
    
    else:
        raise ValueError(f"Path {path} does not exist")

def load_results(input_file: Path) -> list[AnyBURLRule]:
    with open(input_file, 'r') as f:
        lines = f.readlines()

    results: list[AnyBURLRule] = []
    skipped = 0

    for line in tqdm(lines, desc="Loading rules"):
        line = line.strip()
        if not line:
            skipped += 1
            continue

        parts = line.split('\t')
        if len(parts) < 4:
            print(f"Warning: skipping malformed line (expected 4+ parts): {line}")
            skipped += 1
            continue

        score1, score2, confidence = parts[0], parts[1], parts[2]
        rule_string = '\t'.join(parts[3:])

        if "<=" not in rule_string:
            print(f"Warning: skipping malformed rule (missing '<='): {rule_string}")
            skipped += 1
            continue

        head, body_str = [part.strip() for part in rule_string.split("<=", 1)]
        body = [b.strip() for b in body_str.split(",") if b.strip()]
        results.append(AnyBURLRule(int(score1), int(score2), float(confidence), head, body))

    if skipped:
        print(f"Skipped {skipped} lines while loading rules")
    return results

def replace_object_ids_with_name_in_file(rules: list[AnyBURLRule], object_id_dict: dict) -> list[AnyBURLRule]:
    """Replace object IDs with names in a rule file.
    Format: score1\tscore2\tconfidence\trule_string
    Rules contain object IDs like: predicate(X,obj_id)
    """

    modified_rules: list[AnyBURLRule] = []
    skipped = 0
    
    for rule in tqdm(rules, desc="Processing rules"):
        rule_string = rule.get_rule_string()
        
        # Replace object IDs that appear as predicate arguments
        arg_id_pattern:Pattern(str) = re.compile(r'(?<=\(|,)\s*(\d+)\s*(?=,|\))')
        object_ids: list[str] = arg_id_pattern.findall(rule_string)

        replacement_found = False

        def replace_id(match: re.Match) -> str:
            nonlocal replacement_found
            obj_id = int(match.group(1))
            if obj_id in object_id_dict:
                obj_name = object_id_dict[obj_id]
                if isinstance(obj_name, list):
                    obj_name = obj_name[0] if obj_name else ""
                obj_name = str(obj_name).replace(" ", "_")
                replacement_found = True
                return obj_name
            print(f"value not found in object_id_dict {obj_id}")
            return match.group(0)

        modified_rule = arg_id_pattern.sub(replace_id, rule_string)
        # Only write if at least one replacement was made
        if not replacement_found and object_ids:
            skipped += 1
            continue
        
        if "<=" in modified_rule:
            head, body_str = [part.strip() for part in modified_rule.split("<=", 1)]
            body = [b.strip() for b in body_str.split(",") if b.strip()]
            modified_rules.append(AnyBURLRule(rule.score1, rule.score2, rule.confidence, head, body))
        else:
            skipped += 1

    print(f"Processed {len(modified_rules)} rules (skipped {skipped})")
    return modified_rules


def save_rules_to_file(rules: list[AnyBURLRule], output_file: Path) -> None:
    result_lines = [
        f"{rule.score1}\t{rule.score2}\t{rule.confidence}\t{rule.get_rule_string()}\n"
        for rule in rules
    ]
    with open(output_file, 'w') as f:
        f.writelines(result_lines)
    print(f"Output saved to {output_file}")

def remove_low_confidence_rule(rules: list[AnyBURLRule], confidence_threshold: float = 0.5) -> list[AnyBURLRule]:
    return [rule for rule in rules if rule.confidence >= confidence_threshold]

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
        rules = load_results(input_path)
        modified_rules = replace_object_ids_with_name_in_file(rules=rules, object_id_dict=object_id_dict)
        save_rules_to_file(modified_rules, output_path)
    except Exception as e:
        print(f"Exception: {e}")
        
if __name__ == "__main__":
    main()
