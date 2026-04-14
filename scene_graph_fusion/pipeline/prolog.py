from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pydantic import JsonValue
from tqdm import tqdm
import jsonlines
import string


class ProLogFact():
    def __init__(self,predicate:str, subject:str, obj:str|int|None):
        """Initialise a Prolog fact with a predicate, subject, and object."""
        self.predicate = predicate.strip().replace(" ", "_") if predicate else predicate
        self.subject = subject.strip().replace(" ", "_") if subject else subject
        self.obj = str(obj).strip().replace(" ", "_") if obj is not None else obj
    def __str__(self):
        """Return the ProLog fact as a string, e.g. 'pred(subj, obj).'"""
        return f"{self.predicate}({self.subject}, {self.obj})."
    def __eq__(self, other):
        """Check equality of two ProLogFact instances based on their string representation."""
        if isinstance(other, ProLogFact):
            return str(self) == str(other)
        return False
    def __hash__(self):
        return hash(str(self))
    def arguments(self):
        """Return the arguments of the ProLog fact as a tuple (subject, object)."""
        return (self.subject, self.obj)
    def is_complete(self):
        """Check if the ProLog fact has both subject and object defined."""
        return self.subject is not None and self.obj is not None and self.predicate is not None
    def is_intransitive(self):
        """Check if the ProLog fact is intransitive (i.e. has no object)."""
        return self.obj is None and self.subject is not None and self.predicate is not None
    def lower(self):
        """Return a new ProLogFact with all components lowercased."""
        return ProLogFact(
            self.predicate.lower() if self.predicate else self.predicate,
            self.subject.lower() if self.subject else self.subject,
            self.obj.lower() if self.obj else self.obj
        )
    def add_suffix(self, suffix:str):
        """Apply a suffix to the subject and object of the ProLogFact."""
        self.subject = f"{self.subject}_{suffix}" if self.subject else self.subject
        self.obj = f"{self.obj}_{suffix}" if self.obj else self.obj

    @classmethod
    def from_string(cls, fact_str:str) -> ProLogFact | None:
        """Parse a ProLog fact string (e.g. 'pred(subj, obj).') into a ProLogFact instance.

        Returns ``None`` for blank lines and lines whose first non-whitespace
        character is ``%`` (a Prolog comment).
        """
        fact_str = fact_str.strip()
        if not fact_str or fact_str.startswith('%'):
            return None
        fact_str = fact_str.rstrip('.')
        predicate_part, args_part = fact_str.split('(', 1)
        args_part = args_part.rstrip(')')
        subject, obj = [arg.strip() for arg in args_part.split(',', 1)]
        return cls(predicate_part.strip(), subject, obj)
    
class ProLogBiasFact(ProLogFact):
    def __init__(self,predicate:str,arity:int, head:bool):
        """Initialise a Popper bias fact marking a predicate as head or body."""
        if head:
            super().__init__("head_pred", predicate, arity)
        else:
            super().__init__("body_pred", predicate, arity)

class ProLogExample(ProLogFact):
    def __init__(self, predicate:str, subject:str, obj:str, positive:bool):
        """Initialise a Prolog example fact with a label for positive or negative."""
        super().__init__(predicate, subject, obj)
        self.positive = positive
    def __str__(self):
        """Return the Prolog example as a string with label, e.g. 'pos(pred(subj, obj)).'"""
        label = "pos" if self.positive else "neg"
        return f"{label}({self.predicate}({self.subject}, {self.obj}))."
    @classmethod
    def from_string(cls, example_str:str) -> ProLogExample | None:
        """Parse a ProLog example string (e.g. 'pos(pred(subj, obj)).') into a ProLogExample instance.

        Returns ``None`` for blank lines and Prolog comment lines (``%``).
        """
        example_str = example_str.strip()
        if not example_str or example_str.startswith('%'):
            return None
        example_str = example_str.rstrip('.')
        label, rest = example_str.split('(', 1)
        if label not in ('pos', 'neg'):
            raise ValueError(f"Invalid example label: {label}")
        rest = rest.rstrip(')')
        predicate_part, args_part = rest.split('(', 1)
        args_part = args_part.rstrip(')')
        subject, obj = [arg.strip() for arg in args_part.split(',', 1)]
        return cls(predicate_part.strip(), subject, obj, positive=(label == 'pos'))

class ProLogRule():
    def __init__(self, head:ProLogFact, body:list[ProLogFact]):
        """Initialise a Prolog rule with a head and a list of body facts."""
        self.head = head
        self.body = body
    def __str__(self):
        """Return the Prolog rule as a string, e.g. 'head_pred(subj, obj) :- body_pred1(subj, obj), body_pred2(subj, obj).'"""
        body_str = ", ".join(str(fact)[:-1] for fact in self.body)  # remove trailing '.' from body facts
        return f"{str(self.head)[:-1]} :- {body_str}."  # remove trailing '.' from head





class PopperScenario:
    """Facts from a single Popper experiment folder (bk.pl, exs.pl, bias.pl)."""

    def __init__(self, folder_name: str, bk: list[ProLogFact] | None = None,
                 exs: list[ProLogExample] | None = None,
                 bias: list[ProLogBiasFact] | None = None):
        self.folder_name = folder_name
        self.bk: list[ProLogFact] = bk or []
        self.exs: list[ProLogExample] = exs or []
        self.bias: list[ProLogBiasFact] = bias or []

    @classmethod
    def from_folder(cls, folder_path: str) -> PopperScenario:
        """Load a scenario from an existing experiment directory."""
        folder_name = os.path.basename(folder_path)
        bk = _load_if_exists(os.path.join(folder_path, "bk.pl"))
        exs = _load_if_exists(os.path.join(folder_path, "exs.pl"))
        bias = _load_if_exists(os.path.join(folder_path, "bias.pl"))
        return cls(folder_name, bk, exs, bias)
    
    def save(self, output_root: str, backup: bool = False) -> None:
        """Write bk.pl, exs.pl and bias.pl into ``output_root/<folder_name>/``."""
        folder_path = os.path.join(output_root, self.folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for file_name, facts in (("bk.pl", self.bk), ("exs.pl", self.exs), ("bias.pl", self.bias)):
            path = os.path.join(folder_path, file_name)
            if backup:
                _backup_if_needed(path)
            create_popper_file(facts, path)

    def entities(self) -> set[str]:
        """Return all unique subjects/objects from bk and exs facts."""
        ents: set[str] = set()
        for fact in self.bk + self.exs:
            if fact is None:
                continue
            if fact.subject:
                ents.add(fact.subject)
            if fact.obj:
                ents.add(fact.obj)
        return ents
    def replace_entity(self, map_:dict) -> None:
        """Replace all occurrences of *old_entity* with *new_entity* in bk, exs, and bias facts."""
        for fact in self.bk + self.exs + self.bias:
            if fact.subject in map_:
                fact.subject = map_[fact.subject]
            if fact.obj in map_:
                fact.obj = map_[fact.obj]
    def get_bias_head_predicates(self) -> set[str]:
        """Return the set of positive (head_pred) predicate names from the scenario's bias."""
        return {
            fact.subject
            for fact in self.bias
            if fact.predicate == "head_pred"
        }
    def __repr__(self) -> str:
        return f"PopperScenario({self.folder_name!r}, bk={len(self.bk)}, exs={len(self.exs)}, bias={len(self.bias)})"


class PopperDataset:
    """A collection of :class:`PopperScenario` instances loaded from a folder tree."""

    def __init__(self, folder_path: str | None = None):
        self.scenarios: list[PopperScenario] = []
        self.folder_path = folder_path

    def load(self, folder_path: str | None = None) -> None:
        """Load all experiment sub-directories as PopperScenario instances."""
        if folder_path is None:
            folder_path = self.folder_path
        if folder_path is None:
            raise ValueError("No folder path provided for loading Popper dataset.")
        self.folder_path = folder_path
        subdirs = sorted(
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        )
        self.scenarios = [
            PopperScenario.from_folder(os.path.join(folder_path, d))
            for d in tqdm(subdirs, desc="Loading Popper datasets")
        ]
        return self.scenarios

    def save(self, output_folder: str | None = None, backup: bool = False) -> None:
        """Save every scenario to the specified output folder."""
        if output_folder is None:
            output_folder = self.folder_path
        if output_folder is None:
            raise ValueError("No output folder path provided for saving Popper dataset.")
        for scenario in tqdm(self.scenarios, desc="Saving Popper datasets"):
            scenario.save(output_folder, backup=backup)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __iter__(self):
        return iter(self.scenarios)

    def __getitem__(self, idx: int) -> PopperScenario:
        return self.scenarios[idx]

def create_popper_files(values:list[tuple[str,list[ProLogFact]]],output_folder_path:str,file_name:str):
    """Create a Popper .pl file for each image, within the specified output folder + the string passed in values, containing the provided ProLog facts.

    Args:
        values (list[tuple[str,list[ProLogFact]]]): A list of tuples, where each tuple contains a file name and a list of ProLogFact instances.
        output_folder_path (str): The path to the folder where the Popper .pl files will be created.
        file_name (str): The name of the Popper .pl file to create for each image.
    """
    for id_, facts in tqdm(values):
        folder_path = os.path.join(output_folder_path, get_id_from_file_name(id_))
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, file_name)
        create_popper_file(facts, file_path)

def create_popper_file(facts:list[ProLogFact], output_file_path:str):
    """Create a single Popper .pl file containing the provided ProLog facts.

    Args:
        facts (list[ProLogFact]): A list of ProLogFact instances to write to the file.
        output_file_path (str): The path to the Popper .pl file to create.
    """
    with open(output_file_path, 'w') as f:
        facts.sort(key=lambda x: str(x))
        for fact in facts:
            f.write(str(fact) + "\n")


def load_popper_facts(file_path: str) -> list[ProLogFact]:
    """Load Popper facts from a .pl file and return them as a list of ProLogFact or ProLogExample instances."""
    facts = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            try:
                if line.startswith('pos(') or line.startswith('neg('):
                    facts.append(_string_to_example(line))
                else:
                    facts.append(_string_to_fact(line))
            except ValueError:
                print(f"Warning: Could not parse line: {line}")
    return facts

def load_popper_file_dataset(folder_path: str) -> list[PopperScenario]:
    """Load all Popper experiment directories under *folder_path*.

    Each sub-directory is expected to contain ``bk.pl``, ``exs.pl``, and
    ``bias.pl``.  Missing files are silently skipped (empty list).

    Returns:
        A list of :class:`PopperScenario` instances.
    """
    ds = PopperDataset(folder_path)
    ds.load()
    return ds.scenarios


def _load_if_exists(path: str) -> list[ProLogFact]:
    """Load Popper facts from *path* if it exists, otherwise return an empty list."""
    if os.path.exists(path):
        return load_popper_facts(path)
    return []


def _backup_if_needed(path: str) -> None:
    """Copy *path* to a sibling ``*_backup.pl`` file if no backup exists yet."""
    backup = path.replace(".pl", "_backup.pl")
    if not os.path.exists(backup) and os.path.exists(path):
        shutil.copy2(path, backup)


def _string_to_fact(line: str) -> ProLogFact:
    """Parse a Prolog fact string (e.g. 'pred(subj, obj).') into a ProLogFact."""
    line = line.rstrip('.')
    predicate_part, args_part = line.split('(', 1)
    args_part = args_part.rstrip(')')
    subject, obj = [arg.strip() for arg in args_part.split(',', 1)]
    return ProLogFact(predicate_part.strip(), subject, obj)

def _string_to_example(line: str) -> ProLogExample:
    """Parse a Prolog example string (e.g. 'pos(pred(subj, obj)).') into a ProLogExample."""
    line = line.rstrip('.')
    label, rest = line.split('(', 1)
    if label not in ('pos', 'neg'):
        raise ValueError(f"Invalid example label: {label}")
    rest = rest.rstrip(')')
    predicate_part, args_part = rest.split('(', 1)
    args_part = args_part.rstrip(')')
    subject, obj = [arg.strip() for arg in args_part.split(',', 1)]
    return ProLogExample(predicate_part.strip(), subject, obj, positive=(label == 'pos'))


def load_words_from_csv(words_csv_path:str)->list[str]:
    """Load and return all words from a CSV file as a flat lowercased list."""
    words = []
    with open(words_csv_path.strip(), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader,desc="Loading words from CSV"):
            for cell in row:
                word = cell.strip().lower()
                words.append(word)
    return words


def load_word_json(path)->list[dict]:
    """Load a word JSON file and return its 'images' list."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data['images']

def get_id_from_file_name(file_name:str)->str:
    """Extract the numeric image ID from a COCO-style file name (e.g. 'COCO_val2014_000000001234.jpg') --> '000000001234'."""
    if file_name.isdigit():
        return file_name
    assert len(os.path.splitext(file_name)[0].split("_")) >= 3, f"File name {file_name} does not contain enough parts to extract folder name."
    return os.path.splitext(file_name)[0].split("_")[2]





def load_coco_jsonl(coco_jsonl_path:str)->list[dict[str, dict[str]]]:
    """Parse the COCO JSONL annotation file and collect matching verbs and nouns per image."""
    with jsonlines.open(coco_jsonl_path) as reader:
        result =  list()
        for image_file in tqdm(reader, desc="Processing COCO annotations"):
            values = {"file_name": image_file["file_name"], "captions": dict() }
            captions:JsonValue = image_file["captions"]
            
            values["captions"]["scene"] = captions["scene"]
            values["captions"]["action"] = captions["action"]
            values["captions"]["object"] = captions["object"]
            values["captions"]["rationale"] = captions["rationale"]
            result.append(values)
        return result


def load_coco_verbs_nouns_jsonl(coco_jsonl_path: str, verbs: list[str], nouns: list[str]) -> dict[str, list[dict[str, list[str]]]]:
    """Parse the COCO JSONL annotation file and collect matching verbs and nouns per image."""
    translator = str.maketrans('', '', string.punctuation)
    result = {"images": []}
    for image_file in load_coco_jsonl(coco_jsonl_path):
        captions = image_file["captions"]
        phrases = captions["scene"] + captions["action"] + captions["object"]
        found_verbs: set[str] = set()
        found_nouns: set[str] = set()
        for phrase in phrases:
            for word in phrase.lower().split():
                word = word.strip().translate(translator)
                if word in verbs:
                    found_verbs.add(word)
                if word in nouns:
                    found_nouns.add(word)
        result["images"].append({
            "file_name": image_file["file_name"],
            "verbs": list(found_verbs),
            "nouns": list(found_nouns),
        })
    return result
    

    
    
def extract_word_yaml_to_list(data)->list[tuple[str,list[str], list[str]]]:
    """extracts the verbs and nouns and stores them in 2 lists

    Args:
        data (list[dict]): List of dictionaries containing verbs, nouns, and file names.

    Returns:
        list[tuple[str,set[str], set[str]]]: first string contains filename, 1st set contains verbs, second contains nouns
    """
    values = []
    for item in data:
        verbs = set()
        nouns = set()
        for verb in item.get('verbs', []):
            verbs.add(verb)
        for noun in item.get('nouns', []):
            nouns.add(noun)
        file_name = item.get('file_name', '')
        values.append((file_name, list(verbs), list(nouns)))
    return values

def convert_to_prolog_format(relationships:list,objects:list,object_suffix:str="")->list[ProLogFact]:
    """Convert a scene graph's relationship list into a list of ProLogFact instances."""
    prolog_facts = []
    for relationship in relationships:
        subject = relationship['subject']
        predicate = relationship['predicate']
        obj = relationship['object']
        prolog_facts.append(ProLogFact(predicate, objects[subject]['name'] + object_suffix, objects[obj]['name'] + object_suffix))
    return prolog_facts

def load_scene_graph(input_json_path:str, directory:str="/mnt/sda1/Datasets/hl_dataset/scene_graphs/react++")->dict:
    """Load and return a single scene graph JSON file."""
    file_path = os.path.join(directory, input_json_path)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def load_scene_graphs(file_names:list[str]|None, directory:str="/mnt/sda1/Datasets/hl_dataset/scene_graphs/react++")->list[dict]:
    """Load and return multiple scene graph JSON files from the given directory."""
    data_list = []
    if file_names is None:
        file_names = [f for f in os.listdir(directory) if f.endswith('.json')]
        
    for file_name in tqdm(file_names, desc="Loading scene graphs"):
        data_list.append(load_scene_graph(file_name, directory))
    return data_list

def save_to_json(data, output_path):
    """Serialise data to a JSON file at output_path."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def _backup_if_needed(path: str) -> None:
    """Copy *path* to a sibling ``*_backup.pl`` file if no backup exists yet."""
    if os.path.exists(path):
        root,ext = os.path.splitext(path)
        backup = f"{root}_backup{ext}"

        with open(path, "r") as src, open(backup, "w") as dst:
            dst.write(src.read())
            
            
            
#: Default mapping from entity-name prefix (before the numeric suffix) to its
#: Popper semantic type.  Add or override entries to extend the domain.
DEFAULT_CATEGORIES: dict[str, str] = {
    # --- agents: entities that perform actions ---
    "person": "agent",
    "man": "agent",
    "woman": "agent",
    "people": "agent",
    "she": "agent",
    "he": "agent",
    "taxi_driver": "agent",
    "client": "agent",
    "driver": "agent",
    # --- vehicles: self-propelled transport ---
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "bike": "vehicle",
    "bicycle": "vehicle",
    "train": "vehicle",
    "boat": "vehicle",
    "skateboard": "vehicle",
    # --- locations: places / spatial context ---
    "road": "location",
    "street": "location",
    "sidewalk": "location",
    "building": "location",
    "tower": "location",
    "tree": "location",
    "mountain": "location",
    "snow": "location",
    "pole": "location",
    "sign": "location",
    "fence": "location",
    "clock": "location",
    # --- clothing / wearables ---
    "helmet": "clothing",
    "hat": "clothing",
    "jacket": "clothing",
    "shirt": "clothing",
    "pant": "clothing",
    "shoe": "clothing",
    "jean": "clothing",
    # --- parts: structural / mechanical components ---
    "door": "part",
    "window": "part",
    "windshield": "part",
    "tire": "part",
    "wheel": "part",
    "plate": "part",
    "light": "part",
    "logo": "part",
    "wire": "part",
    "leaf": "part",
    "flag": "part",
    "trunk": "part",
    "number": "part",
    # --- body parts ---
    "head": "body_part",
    "hand": "body_part",
    "hair": "body_part",
    # --- animals ---
    "sheep": "animal",
    # --- explicit unknown / placeholder ---
    "none": "entity",
}


class PopperTyper:
    """Infers Popper ``type/2`` declarations from a :class:`~src.pipeline.prolog.PopperScenario`.

    Analyses the background knowledge and examples of a scenario to produce:

    1. A ``type(predicate, (arg_type1, arg_type2)).`` fact for every predicate,
       based on the observed categories of its arguments.
    2. A ``type(constant, (category,)).`` fact for every unique entity constant.

    Argument types are determined per-position across all occurrences of a
    predicate.  If all observations at a given position share the same category,
    that category is used; otherwise the position falls back to ``"entity"``
    (the universal supertype).

    Attributes:
        categories: Maps entity-name prefix → semantic type.  Modify to extend
            or override the default categorisation for your domain.
    """

    def __init__(self, categories: dict[str, str] | None = None) -> None:
        self.categories: dict[str, str] = dict(DEFAULT_CATEGORIES)
        if categories:
            self.categories.update(categories)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def entity_type(self, entity: str) -> str:
        """Return the semantic type for an entity constant.

        Strips the trailing numeric suffix (e.g. ``_000000013274``) before
        looking up the token.  Unknown tokens fall back to ``"entity"``.

        Examples::

            typer.entity_type("bus_000000004140")   # -> "vehicle"
            typer.entity_type("taxi_driver_000021") # -> "agent"
            typer.entity_type("none_000000030508")  # -> "entity"
        """
        token = entity.rsplit("_", 1)[0] if "_" in entity else entity
        return self.categories.get(token.lower(), "entity")

    def infer(self, scenario: PopperScenario) -> list[ProLogFact]:
        """Infer type and constant declarations for *scenario*.

        Returns:
            Sorted, deduplicated list of :class:`ProLogFact` instances containing:

            * ``type(predicate,(t1,t2)).`` — one per predicate in bk/exs.
            * ``constant(entity, category).`` — one per unique entity constant.
        """
        all_facts = [f for f in scenario.bk + scenario.exs if f is not None]
        combined = self._predicate_type_facts(all_facts) + self._constant_type_facts(all_facts)
        return sorted(set(combined), key=str)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predicate_type_facts(self, facts: list[ProLogFact]) -> list[ProLogFact]:
        """Produce one ``type(predicate,(t1,t2)).`` fact per predicate.

        Collects observed argument-position categories across all facts, then
        resolves each position to the unanimous type or ``"entity"`` if mixed.
        """
        observed: dict[str, list[set[str]]] = defaultdict(lambda: [set(), set()])
        for fact in facts:
            if fact.subject:
                observed[fact.predicate][0].add(self.entity_type(fact.subject))
            if fact.obj is not None:
                observed[fact.predicate][1].add(self.entity_type(str(fact.obj)))

        result: list[ProLogFact] = []
        for predicate, (subj_types, obj_types) in sorted(observed.items()):
            arg_types = [self._resolve(subj_types)]
            if obj_types:
                arg_types.append(self._resolve(obj_types))
            result.append(self._make_type_fact(predicate, arg_types))
        return result

    def _constant_type_facts(self, facts: list[ProLogFact]) -> list[ProLogFact]:
        """Produce one ``constant(entity, category).`` fact per unique entity constant."""
        seen: set[str] = set()
        result: list[ProLogFact] = []
        for fact in facts:
            candidates = [fact.subject]
            if fact.obj is not None:
                candidates.append(str(fact.obj))
            for entity in candidates:
                if entity and entity not in seen:
                    seen.add(entity)
                    result.append(ProLogFact("constant", entity, self.entity_type(entity)))
        return result

    @staticmethod
    def _resolve(types: set[str]) -> str:
        """Return the unanimous type, or ``"entity"`` if the set contains multiple."""
        return next(iter(types)) if len(types) == 1 else "entity"

    @staticmethod
    def _make_type_fact(subject: str, arg_types: list[str]) -> ProLogFact:
        """Build a ``type(predicate,(t1,t2)).`` :class:`ProLogFact`.

        Single-argument declarations use a trailing comma to match Popper's
        expected ``(type,)`` syntax, e.g. ``type(on,(entity,vehicle))``.
        """
        if len(arg_types) == 1:
            obj_str = f"({arg_types[0]},)"
        else:
            obj_str = f"({','.join(arg_types)})"
        return ProLogFact("type", subject, obj_str)
