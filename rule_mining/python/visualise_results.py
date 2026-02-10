import random
from pathlib import Path
import argparse
import networkx as nx
import sys
import hashlib
import matplotlib.pyplot as plt  # ty:ignore[unresolved-import]
description=""" Visualises txt result Files, 
    
    """
def main():
    parser = argparse.ArgumentParser(prog="Visualiser",description=description)
    parser.add_argument("filename",help="the file to load")
    parser.add_argument("-n","--num",help="set a maximum number of lines to read (for big files)",type=int)


    args:argparse.Namespace = parser.parse_args()

    
    G = nx.Graph()

    file_path = Path(args.filename)
    if not file_path.absolute().resolve().exists():
        print(f"file '{file_path.absolute()}' not found")
        sys.exit()
    # get file
    num_lines_read = 0
    with open(file_path) as file:
        for line in file:
            if args.num is not None and num_lines_read >= args.num:
                break
            num_lines_read +=1
            sections:list[str] = line.split()
            
            
            
    # print("drawing image")
    
    
    