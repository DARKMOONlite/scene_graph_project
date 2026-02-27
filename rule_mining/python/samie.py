



from .database import RuleDB,Atom,ContextualAtom,read_tsv_file,tsv_to_rules,tsv_to_contextual_rules,Rule
from pathlib import Path
from queue import Queue
from enum import Enum

class Operator(Enum):
    DANGLING_ATOM = "dangling_atom"
    CLOSING_ATOM = "closing_atom"
    INSTANTIATED_ATOM = "instantiated_atom"

class AMIE():
    knowledge_base:RuleDB
    max_rule_length:int = 5
    def __init__(self,knowledge_base:RuleDB) -> None:
        self.knowledge_base = knowledge_base
    
    def mine_rules(self,rules:Queue[Rule],min_support:int=10,min_confidence:float=0.5):
        """in this function we will iteratively generate new rules from the initial set of rules, and prune them based on their support and confidence until we have a final set of rules that are all closed and meet the minimum support and confidence thresholds.

        Args:
            rules (Queue[Rule]): initial set of rules to start mining from, these should be predicates that we want to mine how to create
            min_support (int, optional): minimum support threshold for the rules. Defaults to 10.
            min_confidence (float, optional): minimum confidence threshold for the rules. Defaults to 0.5.

        Returns:
            list[Rule]: final set of rules that are all closed and meet the minimum support and confidence thresholds.
        """
        result:list[Rule] = [] 
        while rules.qsize() > 0: 
            rule = rules.get()
            if rule.is_closed() and not rule.is_pruned(): # if the rule isnt pruned and is closed.
                result.append(rule)
                continue
            # new_rules = self.prune_rules(self.generate_new_rules(rule),rule,min_support,min_confidence)
            for operator in Operator:
                new_rules = self.prune_rules(self.generate_new_rules(rule,operator.value),rule)
                for new_rule in new_rules:
                    rules.put(new_rule)
        return result
        
    def prune_rules(self,rules:list[Rule],original_rule:Rule,min_coverage:float=0.01)->list[Rule]:
        """Prune the given rules based on their support and confidence.

        Args:
            rules (list[Rule]): list of rules to be pruned.
            original_rule (Rule): the original rule from which the new rules were generated.
            min_coverage (float, optional): minimum coverage threshold for the rules. Defaults to 0.01.
        Returns:
            list[Rule]: list of rules that meet the minimum coverage and confidence thresholds.
        """
        result:list[Rule] = []
        confidence_threshold = self.knowledge_base.calculate_pca_confidence(original_rule)
        for rule in rules:
            if len(rule) > self.max_rule_length or \
                self.knowledge_base.calculate_pca_confidence(rule) < confidence_threshold or \
                self.knowledge_base.calculate_head_converage(rule) < min_coverage: #prune if coverage is less than the minimum coverage threshold
                rule.prune()

            result.append(rule)
        return result
    
    def generate_new_rules(self,rule:Rule,operator:str)->list[Rule]:
        """Generates new rules using established mining operators (e.g. add dangling atom, add closing atom, add instantiated atom) on the given rule.
        
        Args:
            rule (Rule): the rule we're going to append new atoms to in order to generate new rules
            operator (str): the operator type to use for generating new rules
        Returns:
            list[Rule]: list of newly generated rules based on the given rule, each should contain 1 more atom than the given rule
        """
        match operator:
            case Operator.DANGLING_ATOM.value:
                return self._generate_dangling_atom_rules(rule)
            case Operator.CLOSING_ATOM.value:
                return self._generate_closing_atom_rules(rule)
            case Operator.INSTANTIATED_ATOM.value:
                return self._generate_instantiated_atom_rules(rule)
            case _:
                return []
    
    

    def _generate_dangling_atom_rules(self,rule:Rule)->list[Rule]:
        """Add Dangling Atom (OD): adds a new atom with one fresh variable and
        one variable shared with the existing rule, for every predicate in the KB.

        For each predicate p and each existing variable v in the rule, generates:
          - p(v, fresh)
          - p(fresh, v)

        Args:
            rule (Rule): the rule to extend

        Returns:
            list[Rule]: new candidate rules, each with one additional dangling atom
        """
        new_rules: list[Rule] = []
        predicates = self.knowledge_base.get_distinct_predicates()
        existing_vars = rule.get_all_variables()

        for pred in predicates:
            for var in existing_vars:
                fresh = rule.fresh_variable()
                # p(var, fresh)
                r1 = rule.copy()
                r1.add(Atom(obj=fresh, sub=var, pred=pred))
                new_rules.append(r1)

                # p(fresh, var)
                r2 = rule.copy()
                r2.add(Atom(obj=var, sub=fresh, pred=pred))
                new_rules.append(r2)

        return new_rules

    def _generate_closing_atom_rules(self,rule:Rule)->list[Rule]:
        """Add Closing Atom (OC): adds a new atom where both arguments are
        variables already present in the rule, for every predicate in the KB.

        For each predicate p and each ordered pair of existing variables (v1, v2),
        generates p(v1, v2). v1 and v2 may be the same variable (self-loops are
        possible in KGs) but the resulting atom must not duplicate an existing atom.

        Args:
            rule (Rule): the rule to extend

        Returns:
            list[Rule]: new candidate rules, each with one additional closing atom
        """
        new_rules: list[Rule] = []
        predicates = self.knowledge_base.get_distinct_predicates()
        existing_vars = list(rule.get_all_variables())

        # Collect existing (sub, obj, pred) triples to avoid duplicates
        existing_atoms = {(a.sub, a.obj, a.pred) for a in rule.atoms}

        for pred in predicates:
            for v1 in existing_vars:
                for v2 in existing_vars:
                    if (v1, v2, pred) in existing_atoms:
                        continue
                    r = rule.copy()
                    r.add(Atom(obj=v2, sub=v1, pred=pred))
                    new_rules.append(r)

        return new_rules

    def _generate_instantiated_atom_rules(self,rule:Rule)->list[Rule]:
        """Add Instantiated Atom (OI): adds a new atom where one argument is a
        concrete entity (constant) from the KB and the other is a variable
        shared with the rule.

        For each predicate p and each existing variable v:
          - p(v, entity)  for each distinct entity appearing as object of p
          - p(entity, v)  for each distinct entity appearing as subject of p

        Args:
            rule (Rule): the rule to extend

        Returns:
            list[Rule]: new candidate rules, each with one additional instantiated atom
        """
        new_rules: list[Rule] = []
        predicates = self.knowledge_base.get_distinct_predicates()
        existing_vars = rule.get_all_variables()

        for pred in predicates:
            objects = self.knowledge_base.get_distinct_entities_for_predicate(pred, "object")
            subjects = self.knowledge_base.get_distinct_entities_for_predicate(pred, "subject")

            for var in existing_vars:
                # p(var, entity) — entity is a constant in the object position
                for entity in objects:
                    r = rule.copy()
                    r.add(Atom(obj=entity, sub=var, pred=pred))
                    new_rules.append(r)

                # p(entity, var) — entity is a constant in the subject position
                for entity in subjects:
                    r = rule.copy()
                    r.add(Atom(obj=var, sub=entity, pred=pred))
                    new_rules.append(r)

        return new_rules

    
        
    def projection_queries(self):
        pass






def main():
    path = Path("../datasets/visual_genome/data/named-triplets.tsv")
    database_path = Path("./database/visual_genome.db")
    triplets = tsv_to_rules(read_tsv_file(path=path))
    with RuleDB(database_path) as db:
        # db.create_db()
        # db.insert_triplet(triplets)
        print(db.get_all_by_scene_graph_id(1))
        
if __name__ == "__main__":
    main()
    