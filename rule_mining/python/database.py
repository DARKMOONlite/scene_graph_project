import itertools
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

GROUNDED_PREDICATE_TYPES = [
"touching above",

]




@dataclass
class Atom:
    """Atom (singular predicate) that can be combined into rules

    Returns:
        _type_: _description_
    """
    obj:str
    sub:str
    pred:str
    scene_graph_id:str|None = None
    def print(self)->str:
        return f"{self.obj}\t{self.pred}\t{self.sub}"
    def contains_entity(self,entity:str)->bool:
        return self.obj == entity or self.sub == entity


@dataclass 
class ContextualAtom(Atom):
    grounded:bool=False
    complexity:int=0
    

class Rule():
    """basic generated rules
    """
    atoms:list[Atom]
    _closed_entities:list[str]
    _hanging_entities:list[str]
    _pruned:bool = False
    def __init__(self) -> None:
        self.atoms = []
        self._closed_entities = []
        self._hanging_entities = []
    def __len__(self):
        return len(self.atoms)
    def add(self,atom:Atom):
        """adds a new atom to the rule

        Args:
            atom (Atom): the new atom to add
        """
        self.atoms.append(atom)
        for entity in [atom.obj, atom.sub]:
            if entity in self._closed_entities:
                continue
            if entity in self._hanging_entities:
                self._hanging_entities.remove(entity)
                self._closed_entities.append(entity)
            else:
                self._hanging_entities.append(entity)
    def is_closed(self)->bool:
        """checks if the rule is closed (i.e. has no hanging entities)

        Returns:
            bool: True if the rule is closed, False otherwise
        """
        return len(self._hanging_entities) == 0
    def get_closed_entities(self)->list[str]:
        """returns the closed entities in the rule

        Returns:
            list[str]: list of closed entities in the rule
        """
        return self._closed_entities
    def get_hanging_entities(self)->list[str]:
        """returns the hanging entities in the rule

        Returns:
            list[str]: list of hanging entities in the rule
        """
        return self._hanging_entities
    def get_non_head_variables(self)->list[str]:
        """returns the variables in the body of the rule (i.e. all atoms except the first one)

        Returns:
            list[str]: list of variables in the body of the rule
        """
        non_head_variables = set()
        for atom in self.atoms[1:]:
            non_head_variables.add(atom.obj)
            non_head_variables.add(atom.sub)
        non_head_variables.discard(self.atoms[0].obj)
        non_head_variables.discard(self.atoms[0].sub)
        return list(non_head_variables)

        
    def get_head_variables(self)->list[str]:
        """returns the variables in the head of the rule (i.e. the first atom)

        Returns:
            list[str]: list of variables in the head of the rule
        """
        return [self.atoms[0].obj,self.atoms[0].sub]
    
    def get_all_variables(self)->set[str]:
        """returns all variables that appear in any atom of the rule"""
        variables = set()
        for atom in self.atoms:
            variables.add(atom.sub)
            variables.add(atom.obj)
        return variables

    def copy(self)->'Rule':
        """Create a deep copy of this rule by replaying all atom additions."""
        from copy import deepcopy
        new_rule = Rule()
        for atom in self.atoms:
            new_rule.add(deepcopy(atom))
        return new_rule

    def fresh_variable(self)->str:
        """Generate a fresh variable name not used in this rule."""
        existing = self.get_all_variables()
        i = 0
        while f"v{i}" in existing:
            i += 1
        return f"v{i}"
    def is_pruned(self)->bool:
        return self._pruned
    def prune(self):
        self._pruned = True
    def __bool__(self):
        return self._pruned == False
class ScallopRule(Rule):
    rule_name:str|None = None
    def __init__(self) -> None:
        super().__init__()
    
    def __str__(self) -> str:
        return ""
    

def read_tsv_file(path:Path,num_lines:int=-1,seperator:str="\t")->list[tuple[str,...]]:
    result:list[tuple[str, ...]] = []
    with open(path) as f:
        if num_lines != -1:
            for line in itertools.islice(f,num_lines):
                result.append(tuple(line.split(seperator)))
        else:
            for line in f.readlines():
                result.append(tuple(line.split(seperator)))
                
    return result

def tsv_to_rules(tsv:list[tuple[str,...]])->list[Atom]:
    result = []
    for predicate in tsv:
        if len(predicate) < 3:
            continue
        if len(predicate) > 3:
            result.append(Atom(predicate[0],predicate[2],predicate[1],scene_graph_id=predicate[3]))
        else:
            result.append(Atom(predicate[0],predicate[2],predicate[1]))
    return result



def tsv_to_contextual_rules(tsv:list[tuple[str,...]])->list[ContextualAtom]:
    result = []
    for predicate in tsv:
        if predicate[1] in GROUNDED_PREDICATE_TYPES:
            Grounded = True
        else:
            Grounded = False
        if len(predicate) < 3:
            continue
        if len(predicate) > 3:
            result.append(ContextualAtom(predicate[0],predicate[2],predicate[1],grounded=Grounded,scene_graph_id=predicate[3]))
        else:
            result.append(ContextualAtom(predicate[0],predicate[2],predicate[1],grounded=Grounded))
    return result




class RuleDB:
    def __init__(self, db_path: Path) -> None:
        self.con = sqlite3.connect(db_path)
        self.con.row_factory = sqlite3.Row

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "RuleDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _rows_to_triplets(self, rows) -> list[Atom]:
        return [Atom(obj=r["object"], sub=r["subject"], pred=r["predicate"]) for r in rows]
    def create_db(self) -> None:
        cur = self.con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS triplets (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                object  TEXT NOT NULL,
                predicate TEXT  NOT NULL,
                scene_graph_id INTEGER
            )
        """)
        # Indexes on individual columns
        cur.execute("CREATE INDEX IF NOT EXISTS idx_subject   ON triplets (subject)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_object    ON triplets (object)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_predicate ON triplets (predicate)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scene_graph_id ON triplets (scene_graph_id)")
        # Composite indexes for predicate/subject and predicate/object pairs
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_subject ON triplets (predicate, subject)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_object  ON triplets (predicate, object)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sub_obj  ON triplets (subject, object)")
        self.con.commit()
        
    def insert_triplet(self, triplets: list[Atom]) -> None:
        self.con.executemany(
            "INSERT INTO triplets (subject, object, predicate, scene_graph_id) VALUES (?, ?, ?, ?)",
            tqdm([(t.sub, t.obj, t.pred, t.scene_graph_id) for t in triplets])
        )
        self.con.commit()

    def get_by_subject(self, subject: str) -> list[Atom]:
        cur = self.con.execute("SELECT * FROM triplets WHERE subject = ?", (subject,))
        return self._rows_to_triplets(cur.fetchall())

    def get_by_object(self, obj: str) -> list[Atom]:
        cur = self.con.execute("SELECT * FROM triplets WHERE object = ?", (obj,))
        return self._rows_to_triplets(cur.fetchall())

    def get_by_predicate(self, predicate: str) -> list[Atom]:
        cur = self.con.execute("SELECT * FROM triplets WHERE predicate = ?", (predicate,))
        return self._rows_to_triplets(cur.fetchall())

    def get_by_predicate_and_subject(self, predicate: str, subject: str) -> list[Atom]:
        cur = self.con.execute(
            "SELECT * FROM triplets WHERE predicate = ? AND subject = ?", (predicate, subject)
        )
        return self._rows_to_triplets(cur.fetchall())

    def get_by_predicate_and_object(self, predicate: str, obj: str) -> list[Atom]:
        cur = self.con.execute(
            "SELECT * FROM triplets WHERE predicate = ? AND object = ?", (predicate, obj)
        )
        return self._rows_to_triplets(cur.fetchall())
    def get_by_subject_and_object(self, subject: str, obj: str) -> list[Atom]:
        cur = self.con.execute(
            "SELECT * FROM triplets WHERE subject = ? AND object = ?", (subject, obj)
        )
        return self._rows_to_triplets(cur.fetchall())
    def calculate_support(self,rule:Rule)->int:
        """Calculate the AMIE support of the given rule.
        
        supp(B => r(x,y)) := #(x,y) : ∃z1,...,zm : B ∧ r(x,y)
        
        Counts the distinct (x, y) pairs in the head for which there exist
        bindings of all other variables making the entire rule (body ∧ head)
        true in the KB.

        Args:
            rule (Rule): the rule for which to calculate support
            
        Returns:
            int: support of the rule
        """
        query, params = self._build_support_query(rule)
        cur = self.con.execute(query, params)
        return cur.fetchone()[0]

    def _build_support_query(self, rule: Rule) -> tuple[str, list[str]]:
        """Build a SQL query that counts distinct (head_subject, head_object)
        pairs for which all atoms in the rule are satisfied in the KB.

        Each atom gets its own alias of the triplets table (t0 for head, t1, t2, ...
        for body atoms). Shared variable names between atoms produce equality
        join conditions, and each atom's predicate is filtered in the WHERE clause.

        Args:
            rule (Rule): the rule to build the query for

        Returns:
            tuple[str, list[str]]: the SQL query string and its parameters
        """
        aliases = [f"t{i}" for i in range(len(rule.atoms))]

        # Track every (alias, column) position for each variable name
        var_positions: dict[str, list[tuple[str, str]]] = {}
        for i, atom in enumerate(rule.atoms):
            for var, col in [(atom.sub, "subject"), (atom.obj, "object")]:
                var_positions.setdefault(var, []).append((aliases[i], col))

        # Predicate filters
        where_conditions = [f"{aliases[i]}.predicate = ?" for i in range(len(rule.atoms))]
        params = [atom.pred for atom in rule.atoms]

        # Equality conditions for shared variables
        join_conditions: list[str] = []
        for positions in var_positions.values():
            for j in range(1, len(positions)):
                join_conditions.append(
                    f"{positions[0][0]}.{positions[0][1]} = {positions[j][0]}.{positions[j][1]}"
                )

        from_clause = ", ".join(f"triplets {a}" for a in aliases)
        where_clause = " AND ".join(where_conditions + join_conditions)

        # Count distinct head (x, y) pairs
        query = (
            f"SELECT COUNT(*) FROM ("
            f"SELECT DISTINCT {aliases[0]}.subject, {aliases[0]}.object "
            f"FROM {from_clause} "
            f"WHERE {where_clause})"
        )
        return query, params
    def calculate_head_converage(self,rule:Rule)->float:
        """Calculate the head coverage of the given rule.
        
        hc(B => r(x,y)) = supp(B => r(x,y)) / #(x,y) : r(x,y)
        
        The support of the rule divided by the number of distinct (x, y) pairs
        in the KB that have the head predicate.

        Args:
            rule (Rule): the rule for which to calculate head coverage

        Returns:
            float: head coverage of the rule
        """
        support = self.calculate_support(rule)
        head_pred = rule.atoms[0].pred
        cur = self.con.execute(
            "SELECT COUNT(DISTINCT subject || '\\0' || object) FROM triplets WHERE predicate = ?",
            (head_pred,)
        )
        head_size = cur.fetchone()[0]
        if head_size == 0:
            return 0.0
        return support / head_size
    
    def calculate_pca_confidence(self, rule: Rule) -> float:
        """Calculate the PCA confidence of the given rule.

        pcaconf(B => r(x,y)) = supp(B => r(x,y))
                                / #(x,y) : ∃z1,...,zm, y' : B ∧ r(x, y')

        The denominator counts distinct (x, y) pairs that satisfy the body,
        where x also appears as the subject of at least one head-predicate
        fact in the KB (the partial completeness assumption).

        Args:
            rule (Rule): the rule for which to calculate PCA confidence

        Returns:
            float: PCA confidence of the rule
        """
        support = self.calculate_support(rule)
        query, params = self._build_pca_denominator_query(rule)
        cur = self.con.execute(query, params)
        denominator = cur.fetchone()[0]
        if denominator == 0:
            return 0.0
        return support / denominator

    def _build_pca_denominator_query(self, rule: Rule) -> tuple[str, list[str]]:
        """Build a SQL query for the PCA confidence denominator.

        Counts distinct (x, y) pairs where the body is satisfied and there
        exists some y' such that r(x, y') is in the KB.

        Args:
            rule (Rule): the rule to build the query for

        Returns:
            tuple[str, list[str]]: the SQL query string and its parameters
        """
        body_atoms = rule.atoms[1:]
        head = rule.atoms[0]

        if len(body_atoms) == 0:
            return "SELECT 0", []

        aliases = [f"t{i}" for i in range(len(body_atoms))]

        # Track variable positions across body atoms
        var_positions: dict[str, list[tuple[str, str]]] = {}
        for i, atom in enumerate(body_atoms):
            for var, col in [(atom.sub, "subject"), (atom.obj, "object")]:
                var_positions.setdefault(var, []).append((aliases[i], col))

        # Predicate filters for body atoms
        where_conditions = [f"{aliases[i]}.predicate = ?" for i in range(len(body_atoms))]
        params: list[str] = [atom.pred for atom in body_atoms]

        # Equality conditions for shared variables within the body
        join_conditions: list[str] = []
        for positions in var_positions.values():
            for j in range(1, len(positions)):
                join_conditions.append(
                    f"{positions[0][0]}.{positions[0][1]} = {positions[j][0]}.{positions[j][1]}"
                )

        # Resolve head variables (x, y) to their body alias.column
        head_x = head.sub  # x
        head_y = head.obj  # y
        x_ref = var_positions[head_x][0]
        y_ref = var_positions[head_y][0]

        # EXISTS: there is some y' such that r(x, y') is in the KB
        exists_clause = (
            f"EXISTS (SELECT 1 FROM triplets t_pca "
            f"WHERE t_pca.predicate = ? AND t_pca.subject = {x_ref[0]}.{x_ref[1]})"
        )
        params.append(head.pred)

        from_clause = ", ".join(f"triplets {a}" for a in aliases)
        where_clause = " AND ".join(where_conditions + join_conditions + [exists_clause])

        query = (
            f"SELECT COUNT(*) FROM ("
            f"SELECT DISTINCT {x_ref[0]}.{x_ref[1]}, {y_ref[0]}.{y_ref[1]} "
            f"FROM {from_clause} "
            f"WHERE {where_clause})"
        )
        return query, params

    def get_distinct_predicates(self) -> list[str]:
        """Get all distinct predicates in the KB."""
        cur = self.con.execute("SELECT DISTINCT predicate FROM triplets")
        return [r["predicate"] for r in cur.fetchall()]

    def get_distinct_entities_for_predicate(self, predicate: str, position: str) -> list[str]:
        """Get all distinct entities that appear in a given position for a predicate.

        Args:
            predicate: the predicate to query
            position: 'subject' or 'object'

        Returns:
            list of distinct entity strings
        """
        cur = self.con.execute(
            f"SELECT DISTINCT {position} FROM triplets WHERE predicate = ?", (predicate,)
        )
        return [r[position] for r in cur.fetchall()]

    def get_all_by_scene_graph_id(self, scene_graph_id: int) -> list[Atom]:
        cur = self.con.execute("SELECT * FROM triplets WHERE scene_graph_id = ?", (scene_graph_id,))
        return self._rows_to_triplets(cur.fetchall())
    def get_all(self) -> list[Atom]:
        cur = self.con.execute("SELECT * FROM triplets")
        return self._rows_to_triplets(cur.fetchall())

