"""Compare two parsed scene graphs using the scene-graph fusion matcher."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import UUID

from tqdm import tqdm

from scene_graph_project.scene_graph_fusion.pipeline import (
    FusionConfig,
    SceneGraph,
    SceneGraphFusion,
    Standardiser,
    load_scene_graph_json,
)
from scene_graph_project.scene_graph_fusion.pipeline.wordnet import install_wordnet


@dataclass(frozen=True)
class ElementComparison:
    """Counts and Dice similarity for one scene-graph element type."""

    graph_a_count: int
    graph_b_count: int
    shared_count: int

    @property
    def only_in_a(self) -> int:
        return self.graph_a_count - self.shared_count

    @property
    def only_in_b(self) -> int:
        return self.graph_b_count - self.shared_count

    @property
    def similarity(self) -> float:
        total = self.graph_a_count + self.graph_b_count
        return 1.0 if total == 0 else 2 * self.shared_count / total


@dataclass(frozen=True)
class SceneGraphComparison:
    """Fusion-based object and relationship comparison results."""

    objects: ElementComparison
    relationships: ElementComparison

    @property
    def overall(self) -> ElementComparison:
        return ElementComparison(
            graph_a_count=self.objects.graph_a_count + self.relationships.graph_a_count,
            graph_b_count=self.objects.graph_b_count + self.relationships.graph_b_count,
            shared_count=self.objects.shared_count + self.relationships.shared_count,
        )

    @property
    def has_differences(self) -> bool:
        """Whether either graph has unmatched objects or relationships."""
        return self.overall.similarity != 1.0


@dataclass(frozen=True)
class FolderComparison:
    """Aggregate results of comparing matching graphs from two folders."""

    matched_count: int
    graph_a_only_count: int
    graph_b_only_count: int
    changed_count: int
    comparison: SceneGraphComparison


def find_scene_graph_pairs(
    graph_a_path: Path, graph_b_path: Path
) -> tuple[list[tuple[Path, Path]], int, int]:
    """Return matching JSON pairs and counts of JSON files unique to each path."""
    if graph_a_path.is_file() and graph_b_path.is_file():
        return [(graph_a_path, graph_b_path)], 0, 0
    if not graph_a_path.is_dir() or not graph_b_path.is_dir():
        raise ValueError("Both paths must be JSON files or both paths must be folders.")

    graph_a_files = {
        path.relative_to(graph_a_path): path for path in graph_a_path.rglob("*.json")
    }
    graph_b_files = {
        path.relative_to(graph_b_path): path for path in graph_b_path.rglob("*.json")
    }
    matching_paths = sorted(graph_a_files.keys() & graph_b_files.keys())
    return (
        [(graph_a_files[path], graph_b_files[path]) for path in matching_paths],
        len(graph_a_files.keys() - graph_b_files.keys()),
        len(graph_b_files.keys() - graph_a_files.keys()),
    )


def aggregate_comparisons(
    comparisons: Iterable[SceneGraphComparison],
graph_a_only_count: int = 0,
    graph_b_only_count: int = 0,
) -> FolderComparison:
    """Combine individual comparisons and count graphs with differences."""
    comparisons = list(comparisons)
    return FolderComparison(
        matched_count=len(comparisons),
        graph_a_only_count=graph_a_only_count,
        graph_b_only_count=graph_b_only_count,
        changed_count=sum(comparison.has_differences for comparison in comparisons),
        comparison=SceneGraphComparison(
            objects=ElementComparison(
                graph_a_count=sum(
                    comparison.objects.graph_a_count for comparison in comparisons
                ),
                graph_b_count=sum(
                    comparison.objects.graph_b_count for comparison in comparisons
                ),
                shared_count=sum(
                    comparison.objects.shared_count for comparison in comparisons
                ),
            ),
            relationships=ElementComparison(
                graph_a_count=sum(
                    comparison.relationships.graph_a_count for comparison in comparisons
                ),
                graph_b_count=sum(
                    comparison.relationships.graph_b_count for comparison in comparisons
                ),
                shared_count=sum(
                    comparison.relationships.shared_count for comparison in comparisons
                ),
            ),
        ),
    )


def compare_scene_graphs(
    graph_a: SceneGraph,
    graph_b: SceneGraph,
    fusion: SceneGraphFusion,
) -> SceneGraphComparison:
    """Compare graphs using the object matches selected by *fusion*.

    A relationship is shared when its subject and object are fusion-matched
    and its standardised predicate is identical in both graphs.
    """
    object_matches = fusion.match([graph_a, graph_b])
    object_mapping: dict[UUID, UUID] = {
        match.obj_a.uid: match.obj_b.uid for match in object_matches
    }
    graph_b_relationships = Counter(
        (
            relationship.subject_uid,
            relationship.canonical_predicate,
            relationship.object_uid,
        )
        for relationship in graph_b.relationships
    )

    shared_relationships = 0
    for relationship in graph_a.relationships:
        mapped_subject = object_mapping.get(relationship.subject_uid)
        mapped_object = object_mapping.get(relationship.object_uid)
        if mapped_subject is None or mapped_object is None:
            continue
        candidate = (
            mapped_subject,
            relationship.canonical_predicate,
            mapped_object,
        )
        if graph_b_relationships[candidate] > 0:
            shared_relationships += 1
            graph_b_relationships[candidate] -= 1

    return SceneGraphComparison(
        objects=ElementComparison(
            graph_a_count=len(graph_a.objects),
            graph_b_count=len(graph_b.objects),
            shared_count=len(object_matches),
        ),
        relationships=ElementComparison(
            graph_a_count=len(graph_a.relationships),
            graph_b_count=len(graph_b.relationships),
            shared_count=shared_relationships,
        ),
    )


def format_comparison_table(comparison: SceneGraphComparison) -> str:
    """Return the comparison as a Markdown table."""
    rows = (
        ("Objects", comparison.objects),
        ("Relationships", comparison.relationships),
        ("Overall", comparison.overall),
    )
    lines = [
        "| Element | Graph A | Graph B | Shared | Only in A | Only in B | Similarity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {name} | {result.graph_a_count} | {result.graph_b_count} | "
        "{result.shared_count} | {result.only_in_a} | {result.only_in_b} | "
        "{result.similarity:.1%} |".format(name=name, result=result)
        for name, result in rows
    )
    return "\n".join(lines)


def compare_paths(
    graph_a_path: Path,
    graph_b_path: Path,
    standardiser: Standardiser,
    fusion: SceneGraphFusion,
) -> FolderComparison:
    """Compare a pair of JSON files or all relative-path matches in two folders."""
    pairs, graph_a_only_count, graph_b_only_count = find_scene_graph_pairs(
        graph_a_path, graph_b_path
    )
    comparisons = []
    for graph_a_file, graph_b_file in tqdm(pairs):
        graph_a = load_scene_graph_json(graph_a_file, source=str(graph_a_file))
        graph_b = load_scene_graph_json(graph_b_file, source=str(graph_b_file))
        for graph in (graph_a, graph_b):
            standardiser.standardise(graph)
            standardiser.blacklist(graph)
        comparisons.append(compare_scene_graphs(graph_a, graph_b, fusion))
    return aggregate_comparisons(
        comparisons, graph_a_only_count, graph_b_only_count
    )


def format_folder_comparison(folder_comparison: FolderComparison) -> str:
    """Return folder-match counts and aggregate graph comparison as Markdown."""
    summary = "\n".join(
        (
            f"Matched scene graphs: {folder_comparison.matched_count}",
            f"Scene graphs only in graph A: {folder_comparison.graph_a_only_count}",
            f"Scene graphs only in graph B: {folder_comparison.graph_b_only_count}",
            f"Scene graphs with differences: {folder_comparison.changed_count}",
        )
    )
    return f"{summary}\n\n{format_comparison_table(folder_comparison.comparison)}"


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Compare two scene-graph JSON files or matching JSON files in two folders "
            "using fusion-based matching."
        )
    )
    parser.add_argument("graph_a", type=Path, help="Path to the first scene-graph JSON file or folder.")
    parser.add_argument("graph_b", type=Path, help="Path to the second scene-graph JSON file or folder.")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="minimum IoU for an object match (default: 0.3)",)
    parser.add_argument( "--wup-threshold", type=float, default=0.85, help="Wu-Palmer threshold for label standardisation (default: 0.85)",)
    args = parser.parse_args()

    install_wordnet()
    standardiser = Standardiser(wup_threshold=args.wup_threshold)
    fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=args.iou_threshold))
    try:
        result = compare_paths(args.graph_a, args.graph_b, standardiser, fusion)
    except ValueError as error:
        parser.error(str(error))
    print(format_folder_comparison(result))


if __name__ == "__main__":
    main()
