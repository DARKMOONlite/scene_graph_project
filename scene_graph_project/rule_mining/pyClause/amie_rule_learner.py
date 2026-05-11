from clause import Learner, Options
from c_clause import Loader #ty: ignore

path_train = f"/home/sebastian/Documents/scene_graph_project/rule_mining/datasets/visual_genome/train.triplet.txt"
path_rules_output = f"rule_mining/pyClause/results/amie_result.txt"

options = Options()
options.set(param="learner.mode",value="amie")
## example parameters - choose any supported AMIE options under key "raw"
# rule length (head+body atom)
options.set(param="learner.amie.raw.maxad", value=4)
options.set(param="learner.amie.raw.mins", value=4)
# special syntax for enforcing -const to be used as flag
options.set(param="learner.amie.raw.const", value="*flag*")
# rule length (head+body atom) for rules with constants
options.set(param="learner.amie.raw.maxadc", value=3)
# you can also add java vm params like so:
# options.set("learner.amie.java_options", ["-Dfile.encoding=UTF-8"]) # add options to list

learner = Learner(options=options.get("learner"))
learner.learn_rules(path_data=path_train, path_output=path_rules_output)

# loader = Loader(options.get("loader"))
# loader.load_data(data=path_train)
# loader.load_rules(rules=path_rules_output)