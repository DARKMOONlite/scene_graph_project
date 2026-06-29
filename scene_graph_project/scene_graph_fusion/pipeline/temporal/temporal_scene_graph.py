

from dataclasses import dataclass
from uuid import UUID
from copy import deepcopy
from scene_graph_project.scene_graph_fusion.pipeline.models import SceneObject, SceneGraph, Relationship




@dataclass
class Link:
    instances: list[SceneObject]
    scene_graphs: list[SceneGraph] # scene graphs the instances are in.
    class_: str = "unknown" # the class of the instances, e.g. "car", "person", etc.


class TemporalSceneGraph:
    """A scene graph that has been stabalised over time. It contains a list of SceneGraphs and a list of timestamps."""
    def __init__(self, graphs: list[SceneGraph], timestamps: list[float] | None = None):
        self.graphs = graphs
        self.timestamps = timestamps if timestamps is not None else [float(i) for i in range(len(graphs))]
        self.links: list[Link] = [] # links between objects across frames. Each link is a list of SceneObjects that are the same entity across frames.
    @property
    def first_frame(self) -> SceneGraph:
        return self.graphs[0] if self.graphs else None
    @property
    def last_frame(self) -> SceneGraph:
        return self.graphs[-1] if self.graphs else None
    @property
    def num_frames(self) -> int:
        return len(self.graphs)
    @property
    def num_links(self) -> int:
        return len(self.links)
    
    def add_link(self, instances: list[SceneObject], scene_graphs: list[SceneGraph], class_: str = "unknown"):
        self.links.append(Link(instances=instances, scene_graphs=scene_graphs, class_=class_))
    
    def compress(self) -> SceneGraph:
        """Compress all frames into one graph by merging objects connected by links."""
        if not self.graphs:
            return SceneGraph(image_id="compressed", source="temporal_stabaliser")

        compressed_graph = SceneGraph(image_id="compressed", source="temporal_stabaliser")

        representative_uid_by_object_uid: dict[UUID, UUID] = {}
        for link in self.links:
            if not link.instances:
                continue
            representative_uid = link.instances[0].uid
            for instance in link.instances:
                representative_uid_by_object_uid[instance.uid] = representative_uid

        merged_objects: dict[UUID, SceneObject] = {}
        for graph in self.graphs:
            for obj in graph.objects:
                merged_uid = representative_uid_by_object_uid.get(obj.uid, obj.uid)
                existing = merged_objects.get(merged_uid)
                if existing is None:
                    merged_obj = deepcopy(obj)
                    merged_obj.uid = merged_uid
                    merged_objects[merged_uid] = merged_obj
                    continue
                existing.attributes = list(dict.fromkeys(existing.attributes + obj.attributes))
                if obj.confidence > existing.confidence:
                    existing.label = obj.label
                    existing.canonical_label = obj.canonical_label
                    existing.bbox = obj.bbox
                    existing.source = obj.source
                    existing.confidence = obj.confidence

        compressed_graph.objects = list(merged_objects.values())
        seen_relationships: set[tuple[UUID, str, UUID]] = set()
        for graph in self.graphs:
            for rel in graph.relationships:
                subject_uid = representative_uid_by_object_uid.get(rel.subject_uid, rel.subject_uid)
                object_uid = representative_uid_by_object_uid.get(rel.object_uid, rel.object_uid)
                if subject_uid == object_uid:
                    continue
                key = (subject_uid, rel.canonical_predicate, object_uid)
                if key in seen_relationships:
                    continue
                seen_relationships.add(key)
                compressed_graph.add_relationship(
                    Relationship(
                        subject_uid=subject_uid,
                        predicate=rel.predicate,
                        object_uid=object_uid,
                        confidence=rel.confidence,
                        source=rel.source,
                        canonical_predicate=rel.canonical_predicate,
                    )
                )

        return compressed_graph
    
    def lossy_compression(self) -> SceneGraph:
        """Compress all frames into one graph by merging objects connected by links, but only keep the first instance of each object."""
        