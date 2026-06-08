"""Filter scene-graph JSON files by standardising labels and removing blacklisted objects."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from scene_graph_project.scene_graph_fusion.pipeline import (
    SceneGraph,
    Standardiser,
    load_scene_graphs_json,
    save_scene_graph_json,
)
from scene_graph_project.scene_graph_fusion.pipeline.io_formats import collect_scene_graph_files
from scene_graph_project.scene_graph_fusion.pipeline.wordnet import install_wordnet
from tqdm import tqdm
OBJECT_BLACKLIST: set[str] = {
    "tree", 
    "building",
    "banner",
    "window",
    "sky",
    "umbrella",
}




def filter_scene_graph(graph: SceneGraph, standardiser: Standardiser) -> SceneGraph:
    """Standardise *graph* and remove blacklisted objects in-place."""
    standardiser.standardise(graph)
    return standardiser.blacklist(graph)


def output_paths_for(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    graph_count: int,
) -> list[Path]:
    """Map one input file to one or more output files while preserving folder structure."""
    relative_path = input_path.relative_to(input_root)
    if graph_count == 1:
        return [output_root / relative_path]

    paths: list[Path] = []
    for index in range(graph_count):
        paths.append(output_root / relative_path.with_name(f"{relative_path.stem}_{index}.json"))
    return paths


def process_folder(input_root: Path, output_root: Path) -> int:
    """Process all JSON files under *input_root* and write filtered graphs to *output_root*."""
    standardiser = Standardiser(blacklist=OBJECT_BLACKLIST)
    processed_graphs = 0

    for input_path in tqdm(collect_scene_graph_files(input_root), desc="Processing scene graphs"):
        graphs = load_scene_graphs_json(input_path, source=str(input_path))
        destinations = output_paths_for(input_path, input_root, output_root, len(graphs))

        for graph, output_path in zip(graphs, destinations):
            filter_scene_graph(graph, standardiser)
            save_scene_graph_json(graph, output_path)
            processed_graphs += 1

    return processed_graphs


def main() -> None:
    """Command-line entry point for filtering scene-graph JSON folders."""
    parser = ArgumentParser(description="Standardise and blacklist scene-graph JSON files in a folder tree.")
    parser.add_argument("input_root", type=Path, help="Root folder containing scene graph JSON files.")
    parser.add_argument("output_root", type=Path, help="Destination folder for filtered scene graph JSON files.")
    args = parser.parse_args()

    install_wordnet()

    if not args.input_root.is_dir():
        raise NotADirectoryError(f"input folder '{args.input_root}' not found")

    args.output_root.mkdir(parents=True, exist_ok=True)
    processed_graphs = process_folder(args.input_root, args.output_root)
    print(f"filtered {processed_graphs} scene graph(s)")


if __name__ == "__main__":
    main()