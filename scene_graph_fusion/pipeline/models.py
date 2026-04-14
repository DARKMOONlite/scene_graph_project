"""Core data models for scene graph fusion.

Provides BoundingBox, SceneObject, Relationship, and SceneGraph — the
common representation that all source formats are converted into before
standardisation and merging.
"""

from __future__ import annotations

from random import random
import uuid
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates.

    Stored as (x_min, y_min, x_max, y_max).
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # --- constructors ---

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        """Create from (x, y, width, height) format."""
        return cls(x, y, x + w, y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        """Create from centre-x, centre-y, width, height."""
        hw, hh = w / 2, h / 2
        return cls(cx - hw, cy - hh, cx + hw, cy + hh)

    # --- geometry helpers ---

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2

    def intersection_area(self, other: BoundingBox) -> float:
        """Area of the intersection rectangle (0 if no overlap)."""
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)
        return max(0.0, ix_max - ix_min) * max(0.0, iy_max - iy_min)

    def iou(self, other: BoundingBox) -> float:
        """Intersection-over-Union with *other*."""
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union

    def contains(self, other: BoundingBox, threshold: float = 0.9) -> bool:
        """True if *other* is mostly inside *self* (>=threshold of other's area)."""
        if other.area <= 0:
            return False
        return self.intersection_area(other) / other.area >= threshold


@dataclass
class SceneObject:
    """A detected object within a scene graph.

    Attributes:
        label: Raw class label as reported by the source detector.
        canonical_label: Standardised label after language normalisation
            (populated by the Standardiser).
        bbox: Bounding box in pixel coordinates (may be ``None`` for
            sources that don't provide spatial info).
        attributes: Free-form attribute strings (colours, materials, …).
        confidence: Detection confidence from the source model.
        source: Name of the source that produced this object.
        uid: Unique identifier, auto-generated.
    """

    label: str
    bbox: BoundingBox | None = None
    attributes: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""
    canonical_label: str = ""
    uid: int = field(default_factory=lambda: uuid.uuid4())

    def __post_init__(self):
        self.label = self.label.strip().lower()
        if not self.canonical_label:
            self.canonical_label = self.label


@dataclass
class Relationship:
    """A directed relationship (edge) between two SceneObjects.

    Attributes:
        subject_uid: UID of the subject SceneObject.
        predicate: Relationship label (e.g. "on", "holding", "next to").
        object_uid: UID of the object SceneObject.
        canonical_predicate: Standardised predicate after normalisation.
        confidence: Confidence from the source model.
        source: Name of the source that produced this relationship.
    """

    subject_uid: int
    predicate: str
    object_uid: int
    confidence: float = 1.0
    source: str = ""
    canonical_predicate: str = ""

    def __post_init__(self):
        self.predicate = self.predicate.strip().lower()
        if not self.canonical_predicate:
            self.canonical_predicate = self.predicate


@dataclass
class SceneGraph:
    """A full scene graph for a single image from one source.

    Attributes:
        image_id: Identifier for the image (e.g. file name, COCO id).
        source: Name of the originating detector / dataset.
        objects: All detected objects.
        relationships: All detected relationships (edges).
    """

    image_id: str
    source: str = ""
    objects: list[SceneObject] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    # --- convenience look-ups ---

    def object_by_uid(self, uid: str) -> SceneObject | None:
        """Return the object with the given UID, or ``None``."""
        for obj in self.objects:
            if obj.uid == uid:
                return obj
        return None

    def objects_by_label(self, label: str) -> list[SceneObject]:
        """Return all objects whose canonical label matches *label*."""
        label = label.lower()
        return [o for o in self.objects if o.canonical_label == label]

    def relationships_for(self, uid: str) -> list[Relationship]:
        """Return all relationships where *uid* is subject or object."""
        return [r for r in self.relationships
                if r.subject_uid == uid or r.object_uid == uid]

    def add_object(self, obj: SceneObject) -> SceneObject:
        """Append an object and return it."""
        self.objects.append(obj)
        return obj

    def add_relationship(self, rel: Relationship) -> Relationship:
        """Append a relationship and return it."""
        self.relationships.append(rel)
        return rel

    def summary(self) -> dict:
        """Return a compact summary of the graph contents."""
        return {
            "image_id": self.image_id,
            "source": self.source,
            "num_objects": len(self.objects),
            "num_relationships": len(self.relationships),
            "labels": sorted({o.canonical_label for o in self.objects}),
            "predicates": sorted({r.canonical_predicate for r in self.relationships}),
        }
    def visualise(self):
        G = nx.Graph()
        for obj in self.objects:
            G.add_node(obj.uid,label=obj.canonical_label)
        for rel in self.relationships:
            G.add_edge(rel.subject_uid,rel.object_uid,label=rel.predicate)
        self._draw_network(network=G)

    def _draw_network(self,network:nx.Graph):
        
        colours:list[tuple] = []
        for _ in range(len(network.nodes)):
            colours.append((random()*0.8,random(),random()))
            
        nx.draw(G=network,pos=nx.spring_layout(G=network,seed=0),node_color=colours)
        nx.draw_networkx_labels(G=network,pos=nx.spring_layout(G=network,seed=0),labels=nx.get_node_attributes(network,'label'))
        plt.show()
        