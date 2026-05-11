import scallopy
from argparse import ArgumentParser

parser = ArgumentParser("runs a scallop file in the scallop interpreter")
parser.add_argument("file")


args = parser.parse_args()


ctx = scallopy.ScallopContext()
ctx.import_file(args.file)
print(f"loaded file {args.file}, now running")
ctx.run()
print("scallop ran")

# paths = ctx.relation("")