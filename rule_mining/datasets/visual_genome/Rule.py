from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm
import re
from typing import Pattern
@dataclass
class AnyBURLRule:
    rule_applications: int 
    correct_applications: int
    confidence: float # Confidence = corrent/rule_applications
    head: str
    body: list[str]
    def get_rule_string(self) -> str:
        return f"{self.head} <= {', '.join(self.body)}"

@dataclass
class Triplet:
    obj:int
    sub:int
    pred:str
    def print(self)->str:
        return f"{self.obj} {self.pred} {self.sub}"

def load_json_files(path:Path,file_limit:int=-1,multiple_files_allowed:bool=True)->list[dict]:
    """Load JSON files from a single file or all JSON files in a folder."""
    
    
    if path.is_file():
        # Single file
        if path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                return [json.load(f)]
        else:
            raise ValueError(f"File {path} is not a JSON file")
    
    elif path.is_dir():
        if multiple_files_allowed is False:
            raise Exception("loading multiple files not allowed")
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


def load_anyBURL_results(input_file: Path) -> list[AnyBURLRule]:
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
            modified_rules.append(AnyBURLRule(rule.rule_applications, rule.correct_applications, rule.confidence, head, body))
        else:
            skipped += 1

    print(f"Processed {len(modified_rules)} rules (skipped {skipped})")
    return modified_rules


def save_rules_to_file(rules: list[AnyBURLRule], output_file: Path) -> None:
    result_lines = [
        f"{rule.rule_applications}\t{rule.correct_applications}\t{rule.confidence}\t{rule.get_rule_string()}\n"
        for rule in rules
    ]
    with open(output_file, 'w') as f:
        f.writelines(result_lines)
    print(f"Output saved to {output_file}")

def remove_low_confidence_rules(rules: list[AnyBURLRule], confidence_threshold: float = 0.5) -> list[AnyBURLRule]:
    return [rule for rule in rules if rule.confidence >= confidence_threshold]

def remove_low_occurance_rules(rules: list[AnyBURLRule],min_successful_occurances:int):
    return [rule for rule in rules if rule.correct_applications >= min_successful_occurances]

def summarize_structure(value, depth: int = 2):
    """Summarises a dict/list structure to a certain level, including counts."""
    if depth <= 0:
        return type(value).__name__

    if isinstance(value, dict):
        summary = {"__count__": len(value),"__type__":"dict"}
        summary.update({
            key: summarize_structure(val, depth - 1)
            for key, val in value.items()
        })
        return summary

    if isinstance(value, list):
        if not value:
            return {"__count__": 0, "__sample__": []}
        return {
            "__count__": len(value),
            "__type__":"list",
            "__sample__": summarize_structure(value[0], depth - 1),
        }

    return type(value).__name__