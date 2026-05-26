"""I/O helpers for loading scene graphs from common formats.

Each loader returns a :class:`~pipeline.models.SceneGraph` using the
package's canonical data model so the result can be fed straight into the
:class:`~pipeline.standardiser.Standardiser` and
:class:`~pipeline.fusion.SceneGraphFusion` pipeline.

Supported formats
-----------------
* **Generic JSON** — ``load_scene_graph_json``
* **Visual Genome style** — ``load_visual_genome``
* **COCO-style detections** (bounding boxes + category labels, no
  relationships) — ``load_coco_detections``
"""

from __future__ import annotations

import json
from pathlib import Path

from scene_graph_project.scene_graph_fusion.pipeline.models import BoundingBox, Relationship, SceneGraph, SceneObject


# ---------------------------------------------------------------------------
# Generic JSON
# ---------------------------------------------------------------------------

def load_scene_graph_json(
    path: str | Path,
    source: str = "",
) -> SceneGraph:
    """Load a scene graph from a JSON file.

    Expected schema::

        {
          "image_id": "...",
          "objects": [
            {"name": "dog", "bbox": [x_min, y_min, x_max, y_max],
            "attributes": ["brown"], "score": 0.95},
            ...
          ],
          "relationships": [
            {"subject": "o1", "predicate": "on", "object": "o2", optional: "score": 0.9},
            ...
          ]
        }

    *bbox* and *attributes* are optional per-object.
    """
    data = _read_json(path)
    return _dict_to_scene_graph(data, source=source)


def load_scene_graphs_json(
    path: str | Path,
    source: str = "",
) -> list[SceneGraph]:
    """Load a list of scene graphs from a JSON file (top-level list)."""
    data = _read_json(path)
    if isinstance(data, dict):
        data = [data]
    return [_dict_to_scene_graph(d, source=source) for d in data]


# ---------------------------------------------------------------------------
# Visual Genome style
# ---------------------------------------------------------------------------

def load_visual_genome(
    path: str | Path,
    source: str = "visual_genome",
) -> list[SceneGraph]:
    """Load Visual Genome-format scene graphs.

    Expects a list of dicts with ``image_id``, ``objects`` (each having
    ``object_id``, ``names``, and optionally ``x``, ``y``, ``w``, ``h``),
    and ``relationships`` (``relationship_id``, ``subject_id``,
    ``predicate``, ``object_id``).
    """
    data = _read_json(path)
    if isinstance(data, dict):
        data = [data]

    graphs: list[SceneGraph] = []
    for entry in data:
        graph = SceneGraph(
            image_id=str(entry.get("image_id", "")),
            source=source,
        )
        id_to_uid: dict[int, int] = {}

        for obj_data in entry.get("objects", []):
            oid = obj_data.get("object_id", obj_data.get("id", ""))
            names = obj_data.get("names", [obj_data.get("name", "")])
            label = names[0] if names else ""
            bbox = _make_bbox_xywh(obj_data) if _has_xywh(obj_data) else None
            attrs = obj_data.get("attributes", [])
            so = SceneObject(label=label, bbox=bbox, attributes=attrs, source=source)
            id_to_uid[oid] = so.uid
            graph.add_object(so)

        for rel_data in entry.get("relationships", []):
            subj_id = rel_data.get("subject_id", rel_data.get("subject", {}).get("object_id"))
            obj_id = rel_data.get("object_id", rel_data.get("object", {}).get("object_id"))
            predicate = rel_data.get("predicate", "")
            if subj_id in id_to_uid and obj_id in id_to_uid:
                graph.add_relationship(Relationship(
                    subject_uid=id_to_uid[subj_id],
                    predicate=predicate,
                    object_uid=id_to_uid[obj_id],
                    source=source,
                ))
        graphs.append(graph)
    return graphs


# ---------------------------------------------------------------------------
# COCO-style detections (boxes only, no relationships)
# ---------------------------------------------------------------------------

def load_coco_detections(
    path: str | Path,
    categories: dict[int, str] | None = None,
    source: str = "coco",
) -> dict[str, SceneGraph]:
    """Load COCO-style detection results into per-image SceneGraphs.

    *path* should point to a JSON file with a top-level list of dicts::

        [
          {"image_id": 123, "category_id": 1, "bbox": [x, y, w, h],
           "score": 0.92},
          ...
        ]

    *categories* maps ``category_id`` to a label string.  When ``None``,
    ``str(category_id)`` is used.

    Returns a dict mapping ``image_id`` → :class:`SceneGraph`.
    """
    data = _read_json(path)
    if not isinstance(data, list):
        data = data.get("annotations", data.get("detections", []))

    graphs: dict[str, SceneGraph] = {}
    for det in data:
        img_id = str(det["image_id"])
        if img_id not in graphs:
            graphs[img_id] = SceneGraph(image_id=img_id, source=source)

        cat_id = det.get("category_id", det.get("class_id", 0))
        label = (categories or {}).get(cat_id, str(cat_id))
        raw_bbox = det.get("bbox", [])
        bbox = BoundingBox.from_xywh(*raw_bbox) if len(raw_bbox) == 4 else None
        confidence = float(det.get("score", det.get("confidence", 1.0)))

        graphs[img_id].add_object(SceneObject(
            label=label,
            bbox=bbox,
            confidence=confidence,
            source=source,
        ))
    return graphs


# ---------------------------------------------------------------------------
# Dict → SceneGraph conversion
# ---------------------------------------------------------------------------

def _dict_to_scene_graph(data: dict, source: str = "") -> SceneGraph:
    """Convert a generic JSON dict to a SceneGraph."""
    graph = SceneGraph(
        image_id=str(data.get("image_id", "")),
        source=source or data.get("source", ""),
    )
    id_to_uid: dict[int, int] = {}
    iterator = 0
    for obj_data in data.get("objects", []):
        oid = iterator
        
        label = obj_data.get("label", obj_data.get("name", ""))
        raw_bbox = obj_data.get("bbox")
        bbox = BoundingBox(*raw_bbox) if raw_bbox and len(raw_bbox) == 4 else None
        attrs = obj_data.get("attributes", [])
        confidence = float(obj_data.get("score", obj_data.get("confidence", 1.0)))

        so = SceneObject(
            label=label,
            original_identifier=iterator,
            bbox=bbox,
            attributes=attrs,
            confidence=confidence,
            source=source,
        )
        iterator += 1
        id_to_uid[oid] = so.uid
        graph.add_object(so)

    for rel_data in data.get("relationships", []):
        subj_id = int(rel_data.get("subject", -1))
        obj_id = int(rel_data.get("object", -1))
        predicate = rel_data.get("predicate", "")
        confidence = float(rel_data.get("score", rel_data.get("confidence", 1.0)))

        subj_uid = id_to_uid.get(subj_id, subj_id)
        obj_uid = id_to_uid.get(obj_id, obj_id)
        graph.add_relationship(Relationship(
            subject_uid=subj_uid,
            predicate=predicate,
            object_uid=obj_uid,
            confidence=confidence,
            source=source,
        ))
    return graph


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def scene_graph_to_dict(graph: SceneGraph) -> dict:
    """Serialise a SceneGraph to a plain dict suitable for JSON export."""
    uid_to_id: dict[str, str] = {}
    objects = []
    for i, obj in enumerate(graph.objects):
        oid = f"{i}" # uuid is replaced with a simple integer ID for JSON export
        uid_to_id[obj.uid] = oid
        entry: dict = {
            "id": oid,
            "label": obj.canonical_label or obj.label,
            "confidence": obj.confidence,
            "source": obj.source,
        }
        if obj.bbox:
            entry["bbox"] = [obj.bbox.x_min, obj.bbox.y_min, obj.bbox.x_max, obj.bbox.y_max]
        if obj.attributes:
            entry["attributes"] = obj.attributes
        objects.append(entry)

    relationships = []
    for rel in graph.relationships:
        relationships.append({
            "subject_id": uid_to_id.get(rel.subject_uid, rel.subject_uid),
            "predicate": rel.canonical_predicate or rel.predicate,
            "object_id": uid_to_id.get(rel.object_uid, rel.object_uid),
            "confidence": rel.confidence,
            "source": rel.source,
        })

    return {
        "image_id": graph.image_id,
        "source": graph.source,
        "objects": objects,
        "relationships": relationships,
    }


def save_scene_graph_json(graph: SceneGraph, path: str | Path) -> None:
    """Write a SceneGraph to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scene_graph_to_dict(graph), f, indent=2)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _read_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _has_xywh(d: dict) -> bool:
    return all(k in d for k in ("x", "y", "w", "h"))


def _make_bbox_xywh(d: dict) -> BoundingBox:
    return BoundingBox.from_xywh(
        float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])
    )
