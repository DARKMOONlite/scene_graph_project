
import rule_mining.datasets.visual_genome.Rule as rl
import argparse
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="converts logic file to a scallop relationship file")
parser.add_argument("-f","--file",type=str,required=True,help="the folder or specific file that should be loaded")
parser.add_argument("-o","--output",type=str,required=True,help="the file to print the output to")
args:argparse.Namespace = parser.parse_args()


input_path = Path(args.file)
output_path=Path(args.output)

with open(file=input_path,mode="r") as f:
    with open(file=output_path,mode="w") as o:
        lines = f.readlines()
        
        for line in tqdm(lines):
            rule:rl.FOLRule = rl.FOLRule.from_line(line)
            o.write(rule.print_scallop()+"\n")
        





