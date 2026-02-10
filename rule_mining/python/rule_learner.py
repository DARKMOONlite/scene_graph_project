from clause import Learner, Options
from c_clause import Loader  # ty:ignore[unresolved-import]

path_train = f"/home/sebastian/Documents/scene_graph_project/rule_mining/datasets/mini_sets/FB15/train.txt"
path_rules_output = f"rule_mining/python/results/result.txt"

options = Options()

options.set("learner.mode", "anyburl")
# learning time
options.set("learner.anyburl.time", 30)
# set any raw AnyBURL parameter under "... .raw"
# max body atoms of B-rules
options.set("learner.anyburl.raw.MAX_LENGTH_CYCLIC", 5)
# num threads
options.set("learner.anyburl.raw.WORKER_THREADS", 2)
# you can also add java vm params like so:
# options.set("learner.anyburl.java_options", ["-Dfile.encoding=UTF-8"]) # add options to list

learner = Learner(options=options.get("learner"))
learner.learn_rules(path_data=path_train, path_output=path_rules_output)

# load rules with loader
loader = Loader(options.get("loader"))
loader.load_data(data=path_train)
loader.load_rules(rules=path_rules_output)