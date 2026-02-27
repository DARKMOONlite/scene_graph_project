import random
from pathlib import Path
import argparse
import networkx as nx
import sys
import hashlib
import matplotlib.pyplot as plt
description=""" Visualises txt graph Files, 
    they should come in the form of Object Predicate Object, delimited by spaces, with each new predicate being on its own line.
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
            node1:str = hashlib.sha256(string=sections[0].encode()).hexdigest()
            node2:str = hashlib.sha256(string=sections[2].encode()).hexdigest()
            
            print(f"adding node {node1} and {node2}")
            G.add_node(node_for_adding=node1)
            G.add_node(node_for_adding=node2)
            G.add_edge(u_of_edge=node1,v_of_edge=node2,name=sections[1].encode())
            
            
    print("drawing image")
    
    draw_network(network=G)
    
    
    
            
            
            
def draw_network(network:nx.Graph):
    
    colours:list[tuple] = []
    for node in network.nodes:
        colours.append((random.random()*0.8,random.random(),random.random()))
    nx.draw(G=network,pos=nx.spring_layout(G=network,seed=0),node_color=colours)
    plt.show()

if __name__ == "__main__":
    main()