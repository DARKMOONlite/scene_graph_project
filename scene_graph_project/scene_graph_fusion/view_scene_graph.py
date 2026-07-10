from scene_graph_project.scene_graph_fusion.pipeline.io_formats import load_scene_graph_json
from scene_graph_project.scene_graph_fusion.pipeline.models import SceneGraph
from argparse import ArgumentParser
from matplotlib import pyplot as plt





if __name__ == "__main__":
    parser = ArgumentParser(description="View a scene graph JSON file.")
    parser.add_argument("file", type=str, help="Path to the scene graph JSON file.",nargs="+")
    args = parser.parse_args()

    num_files = len(args.file)
    fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))
    if num_files == 1:
        axes = [axes]  # Make it iterable for consistency
    for idx, file in enumerate(args.file):
        sg = load_scene_graph_json(file, source=file)
        ax = axes[idx]
        sg.visualise(axis=ax, show_plot=False)
    plt.show()
        