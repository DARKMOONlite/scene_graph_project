"""scene_graph_fusion.pipeline — scene graph fusion package.

Provides data models, language standardisation, and spatial+semantic
fusion for merging scene graphs from multiple sources.
"""

from .models import BoundingBox, SceneGraph, SceneObject, Relationship, SceneGraphJsonEncoder
from .standardiser import Standardiser
from .fusion import SceneGraphFusion, FusionConfig, ObjectMatch
from .io_formats import (
    load_scene_graph_json,
    load_scene_graphs_json,
    load_visual_genome,
    load_coco_detections,
    save_scene_graph_json,
    scene_graph_to_dict,
)

__all__ = [
    # models
    "BoundingBox",
    "SceneGraph",
    "SceneObject",
    "Relationship",
    # standardisation
    "Standardiser",
    # fusion
    "SceneGraphFusion",
    "ObjectMatch",
    "FusionConfig",
    # I/O
    "load_scene_graph_json",
    "load_scene_graphs_json",
    "load_visual_genome",
    "load_coco_detections",
    "save_scene_graph_json",
    "scene_graph_to_dict",
    "SceneGraphJsonEncoder",
]