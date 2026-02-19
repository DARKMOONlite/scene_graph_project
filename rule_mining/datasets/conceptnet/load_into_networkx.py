"""
Loads a ConceptNet CSV file into a NetworkX directed graph.

Each row becomes a directed edge:  subject  --(relation)--> object
Edge attributes include relation, weight, dataset, and surface text.
Node labels are the full ConceptNet URIs (e.g. /c/en/dog).

Usage examples
--------------
# Load everything, pickle the graph for later reuse
python3 load_into_networkx.py filtered-conceptnet.csv -o graph.pkl

# Load only English concepts for IsA / PartOf, print basic stats
python3 load_into_networkx.py filtered-conceptnet.csv -l en -r /r/IsA /r/PartOf --stats
"""
import ast
import pickle
import sys
from argparse import ArgumentParser, Namespace
from itertools import islice

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def uri_language(uri: str) -> str | None:
    """Return the language code from a concept URI like /c/en/dog → 'en'."""
    parts = uri.split("/")
    return parts[2] if len(parts) >= 3 and parts[1] == "c" else None


def parse_metadata(raw: str) -> dict:
    try:
        return ast.literal_eval(raw.strip())
    except (ValueError, SyntaxError):
        return {}


def load_graph(
    input_file: str,
    languages: set[str] | None,
    relations: set[str] | None,
    chunk_size: int,
) -> nx.DiGraph:
    G = nx.DiGraph()

    with open(input_file) as infile:
        with tqdm(desc="Loading into NetworkX", unit="lines", file=sys.stderr) as pbar:
            while True:
                chunk = list(islice(infile, chunk_size))
                if not chunk:
                    break

                for line in chunk:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 5:
                        continue

                    relation = fields[1].strip()
                    subject  = fields[2].strip()
                    obj      = fields[3].strip()

                    # Relation filter
                    if relations and relation not in relations:
                        continue

                    # Language filter (both endpoints must match)
                    if languages:
                        if uri_language(subject) not in languages:
                            continue
                        if uri_language(obj) not in languages:
                            continue

                    meta    = parse_metadata(fields[4])
                    weight  = meta.get("weight", 1.0)
                    dataset = meta.get("dataset", "")
                    surface = meta.get("surfaceText", "")

                    G.add_edge(
                        subject,
                        obj,
                        relation=relation,
                        weight=weight,
                        dataset=dataset,
                        surface_text=surface,
                    )

                pbar.update(len(chunk))

    return G


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_node(G: nx.DiGraph, node: str) -> None:
    """Plot a node and all of its immediate neighbours in matplotlib."""
    if node not in G:
        print(f"Node '{node}' not found in graph.", file=sys.stderr)
        print("Tip: node names are full URIs, e.g. /c/en/dog", file=sys.stderr)
        return

    # Build a subgraph of the node + all direct neighbours
    neighbours = set(G.predecessors(node)) | set(G.successors(node))
    sub_nodes  = neighbours | {node}
    subG       = G.subgraph(sub_nodes).copy()

    # Shorten URIs for readable labels: /c/en/some_concept → some_concept
    def short(uri: str) -> str:
        parts = uri.split("/")
        if len(parts[-1]) <=2: 
            return parts[-2] if parts[-2] else uri
        return parts[-1] if parts[-1] else uri

    labels = {n: short(n) for n in subG.nodes()}

    # Colour the focal node differently
    node_colours = ["#e74c3c" if n == node else "#3498db" for n in subG.nodes()]

    # Collect edge labels (relation name only)
    edge_labels = {
        (u, v): short(data.get("relation", ""))
        for u, v, data in subG.edges(data=True)
    }

    pos = nx.spring_layout(subG, seed=42, k=2)

    plt.figure(figsize=(14, 9))
    nx.draw_networkx_nodes(subG, pos, node_color=node_colours, node_size=1200)
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(
        subG, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=25,
        edge_color="#7f8c8d",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=20,
        min_target_margin=20,
    )
    nx.draw_networkx_edge_labels(
        subG, pos, edge_labels=edge_labels,
        font_size=7, label_pos=0.35,
    )
    plt.title(f"Neighbourhood of {node}  ({len(sub_nodes)} nodes, {subG.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = ArgumentParser(
        description="Load a ConceptNet CSV into a NetworkX DiGraph."
    )
    parser.add_argument("input_file", help="Path to the ConceptNet CSV file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Pickle the resulting graph to this file (e.g. graph.pkl)",
    )
    parser.add_argument(
        "-l", "--languages",
        nargs="*",
        default=None,
        metavar="LANG",
        help="Keep only edges where both endpoints match these language codes "
             "(e.g. -l en fr). Omit to keep all languages.",
    )
    parser.add_argument(
        "-r", "--relations",
        nargs="*",
        default=None,
        metavar="REL",
        help="Keep only edges with these relations "
             "(e.g. -r /r/IsA /r/PartOf). Omit to keep all.",
    )
    parser.add_argument(
        "-c", "--chunk-size",
        type=int,
        default=50_000,
        help="Lines read per progress-bar tick (default: 50 000)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print basic graph statistics after loading",
    )
    parser.add_argument(
        "-n", "--node",
        default=None,
        metavar="URI",
        help="Plot this node and all its immediate neighbours in matplotlib "
             "(e.g. --node /c/en/dog)",
    )
    args: Namespace = parser.parse_args()

    languages: set[str] | None = set(args.languages) if args.languages else None
    relations: set[str] | None = set(args.relations) if args.relations else None

    print(f"Input:     {args.input_file}", file=sys.stderr)
    print(f"Languages: {languages or 'all'}", file=sys.stderr)
    print(f"Relations: {relations or 'all'}", file=sys.stderr)
    print(file=sys.stderr)

    G = load_graph(args.input_file, languages, relations, args.chunk_size)

    if args.stats:
        print(f"\n--- Graph statistics ---", file=sys.stderr)
        print(f"  Nodes : {G.number_of_nodes():,}", file=sys.stderr)
        print(f"  Edges : {G.number_of_edges():,}", file=sys.stderr)
        if G.number_of_nodes():
            degrees = [d for _, d in G.degree()]
            print(f"  Avg degree : {sum(degrees) / len(degrees):.2f}", file=sys.stderr)

    if args.output:
        with open(args.output, "wb") as f:
            pickle.dump(G, f)
        print(f"\nGraph saved to: {args.output}", file=sys.stderr)

    if args.node:
        visualise_node(G, args.node)

    return G  # useful when imported as a module


if __name__ == "__main__":
    main()
