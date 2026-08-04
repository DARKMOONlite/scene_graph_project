# Scene Graph Fusion

Fuse scene graphs from multiple sources into a single unified graph.

## Approach

1. **Bounding boxes** — determine the rough location of objects and prevent
   distinct instances of the same class being collapsed into one node.
2. **WordNet-based standardisation** — capture meaning via hypernym hierarchies
   so that objects labelled differently by different scene-graph generators
   (e.g. *sofa* vs *couch*) are recognised as equivalent.
3. **Conflict handling** — when spatially-overlapping objects have incompatible
   classes that cannot be reconciled through the hypernym chain, both are kept
   in the merged graph.

## Package structure

```
pipeline/
  models.py         # BoundingBox, SceneObject, Relationship, SceneGraph
  standardiser.py   # Language normalisation (lemmatisation + WordNet)
  fusion.py         # Spatial (IoU) + semantic matching and merging
  io_formats.py     # Loaders for generic JSON, Visual Genome, COCO detections
  wordnet.py        # WordNet hierarchy / Wu-Palmer similarity helpers
  nlp.py            # spaCy-based pronoun resolution and triplet extraction
  prolog.py         # Prolog fact representation (Popper integration)
  database.py       # SQLite storage for intermediate results
```

## Quick start

```python
from pipeline import (
    Standardiser,
    SceneGraphFusion,
    FusionConfig,
    load_scene_graph_json,
    save_scene_graph_json,
)

# 1. Load scene graphs from different sources
sg_a = load_scene_graph_json("detections_model_a.json", source="model_a")
sg_b = load_scene_graph_json("detections_model_b.json", source="model_b")

# 2. Standardise language across both graphs
std = Standardiser(wup_threshold=0.85)
std.standardise(sg_a)
std.standardise(sg_b)

# 3. Fuse into a single graph
fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=0.3))
merged = fusion.fuse([sg_a, sg_b])

# 4. Export the result
save_scene_graph_json(merged, "merged_graph.json")
print(merged.summary())
```

## Comparing two scene graphs

Use the same standardisation and fusion matching configuration to report
shared and unique objects and relationships:

```bash
python3 scene_graph_project/scene_graph_fusion/compare_scene_graphs.py \
  graph_a.json graph_b.json
```
or 

```bash
python3 scene_graph_project/scene_graph_fusion/compare_scene_graphs.py \
  /path/to/graphs_a /path/to/graphs_b
```
The script prints the number of objects/relationships in each graph (or all of them), howmany of each existed in both and how many were seemingly unique to each graph, uses the same `--iou-threshold` and `--wup-threshold` parameters to tune how loose to check.



## Key classes

| Class | Purpose |
|---|---|
| `BoundingBox` | Axis-aligned box with IoU, intersection, containment |
| `SceneObject` | Detected object with label, bbox, attributes, confidence |
| `Relationship` | Directed edge (subject → predicate → object) |
| `SceneGraph` | Full graph for a single image (objects + relationships) |
| `Standardiser` | Lemmatisation + synonym map + WordNet matching |
| `SceneGraphFusion` | Greedy IoU + semantic matching → merged graph |
| `FusionConfig` | Tuneable thresholds for the fusion algorithm |
