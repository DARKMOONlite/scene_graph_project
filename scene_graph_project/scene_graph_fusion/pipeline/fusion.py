"""Scene graph fusion — merge multiple scene graphs into one.

The fusion pipeline:
    1. **Object matching** — pair objects across sources using spatial overlap
       (IoU on bounding boxes) and semantic similarity (canonical labels from
       the Standardiser).
    2. **Object merging** — matched objects are collapsed into a single node,
       keeping the most confident / specific label, and unioning attributes.
    3. **Relationship merging** — relationships are re-pointed at the merged
       nodes; duplicates (same subject, predicate, object) are collapsed.
    4. **Conflict handling** — spatially overlapping objects with incompatible
       labels are kept as separate nodes with a ``conflict`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scene_graph_project.scene_graph_fusion.pipeline.models import BoundingBox, Relationship, SceneGraph, SceneObject
from scene_graph_project.scene_graph_fusion.pipeline.wordnet import wup_confidence
from  dataclasses import astuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FusionConfig:
    """Tunable parameters for the fusion algorithm."""

    iou_threshold: float = 0.3
    """Minimum IoU for two bounding boxes to be considered spatially
    overlapping (candidate match)."""

    semantic_threshold: float = 0.6
    """Minimum Wu-Palmer similarity score (on canonical labels) to accept a
    semantic match between two objects. if set to 0.0, any non-zero similarity is accepted."""

    require_spatial_overlap: bool = True
    """When ``True``, objects from different sources can only be matched if
    their bounding boxes overlap above ``iou_threshold``.  Set to ``False``
    for sources that don't supply bounding boxes."""
    
    #TODO think of better name
    inverse_confidence_calculation: bool = False 
    """When ``True``, the confidence of a merged object is calculated as the
    inverse of the product of the inverse confidences of its constituent objects.
    """
    
    label_set: set[str] | None = None
    """Optional set of allowed canonical labels. If provded, the merged scene_graph will only contain objects whose canonical label is in this set. This can be used to enforce a consistent label space across sources."""
    


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------


@dataclass
class ObjectMatch:
    """A pair of objects from different sources that were matched."""

    obj_a: SceneObject
    obj_b: SceneObject
    iou: float
    semantic_score: float

    def equals(self,other:ObjectMatch) -> bool:
        """True if this match tracks the same base object as *other*."""
        result:bool = \
        self.obj_a.uid == other.obj_a.uid or \
        self.obj_b.uid == other.obj_b.uid  
        
        return result





# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class SceneGraphFusion:
    """Merge multiple :class:`SceneGraph` instances into one unified graph.

    Usage::

        fusion = SceneGraphFusion(config=FusionConfig(iou_threshold=0.4))
        merged = fusion.fuse([graph_a, graph_b, graph_c])
    """

    def __init__(self, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, graphs: list[SceneGraph]) -> list[ObjectMatch]:
        """Find matches between objects in multiple scene graphs."""
        if not graphs:
            return []
        if len(graphs) == 1:
            return []

        all_matches: list[ObjectMatch] = []
        for i in range(len(graphs) - 1):
            for j in range(i + 1, len(graphs)):
                matches, _ = self._match_objects(graphs[i].objects, graphs[j].objects)
                all_matches.extend(matches)
        return all_matches

    def fuse(self, graphs: list[SceneGraph]) -> SceneGraph:
        """Merge a list of scene graphs and return the unified result.

        All input graphs should already have been passed through the
        :class:`~pipeline.standardiser.Standardiser` so that canonical
        labels are populated.
        """
        if not graphs:
            return SceneGraph(image_id="")
        merged = self.merge_scene_graphs(graphs)
            
        if self.config.label_set is not None: # filter out objects whose canonical label is not in the allowed set
            merged.objects = [o for o in merged.objects if o.canonical_label in self.config.label_set]
            # also filter relationships to only include those whose subject and object are still present
            valid_uids = set(o.uid for o in merged.objects)
            merged.relationships = [r for r in merged.relationships if r.subject_uid in valid_uids and r.object_uid in valid_uids]    
        
        return merged

    # ------------------------------------------------------------------
    # Multi-graph merge
    # ------------------------------------------------------------------
    
    def merge_scene_graphs(self, scene_graphs: list[SceneGraph]) -> SceneGraph:
        """Merge all graphs in one pass so group values are order-independent."""
        if not scene_graphs:
            return SceneGraph(image_id="")

        objects = [
            (graph_index, obj)
            for graph_index, graph in enumerate(scene_graphs)
            for obj in graph.objects
        ]
        parents = {obj.uid: obj.uid for _, obj in objects} # links each object to its parent in the union-find structure, this will be updated
        graph_sets = {obj.uid: {graph_index} for graph_index, obj in objects} # links each object to the set of input graphs it represents

        def find(uid):
            """
            finds the root of the union-find structure for the given uid, and performs path compression to speed up future queries
            """
            while parents[uid] != uid:
                parents[uid] = parents[parents[uid]]
                uid = parents[uid]
            return uid

        candidates: list[tuple[float, SceneObject, SceneObject]] = [] # candidates to merge that are above the threshold.
        
        #? compare all objects pairwise, but only consider pairs from different input graphs
        for index, (graph_a_index, obj_a) in enumerate(objects):
            for graph_b_index, obj_b in objects[index + 1:]:
                # An object can only be merged with objects from other inputs.
                if graph_a_index == graph_b_index:
                    continue
                iou = self._compute_iou(obj_a, obj_b)
                semantic_score = self._semantic_similarity(obj_a, obj_b)
                spatial_ok = (
                    not self.config.require_spatial_overlap
                    or iou >= self.config.iou_threshold
                    or obj_a.bbox is None
                    or obj_b.bbox is None
                )
                if spatial_ok and semantic_score >= self.config.semantic_threshold:
                    candidates.append((iou + semantic_score, obj_a, obj_b))


        #? sorts first by the score, then by the object attributes to ensure deterministic ordering. This is important for reproducibility and consistency across runs.
        comparison = lambda candidate: (candidate[0], astuple(candidate[1]), astuple(candidate[2]))  
        for _, obj_a, obj_b in sorted(candidates, key=comparison, reverse=True): # ponytail: global greedy matching
            root_a, root_b = find(obj_a.uid), find(obj_b.uid)
            if root_a == root_b or graph_sets[root_a] & graph_sets[root_b]:
                continue
            parents[root_b] = root_a
            graph_sets[root_a].update(graph_sets[root_b]) # update the graph sets so that we only merge objects from different input graphs. This prevents merging objects that have already been merged with others from the same input graph.

        # Aggregate each connected match group exactly once.
        groups = {}
        for _, obj in objects:
            groups.setdefault(find(obj.uid), []).append(obj)

        uid_remap = {}
        merged_objects = []
        for group in groups.values():
            merged_obj = self._merge_objects(group)
            merged_objects.append(merged_obj)
            uid_remap.update({obj.uid: merged_obj.uid for obj in group})

        relationship_groups = {}
        for graph in scene_graphs:
            for relationship in graph.relationships:
                # Point edges at their merged endpoints before deduplicating.
                key = (
                    uid_remap.get(relationship.subject_uid, relationship.subject_uid),
                    relationship.canonical_predicate,
                    uid_remap.get(relationship.object_uid, relationship.object_uid),
                )
                relationship_groups.setdefault(key, []).append(relationship)

        merged_relationships = [
            Relationship(
                subject_uid=subject_uid,
                predicate=min(relationship.predicate for relationship in relationships),
                object_uid=object_uid,
                canonical_predicate=canonical_predicate,
                confidence=self._merge_confidences([relationship.confidence for relationship in relationships]),
                source=_join_sources(
                    *(relationship.source for relationship in relationships)
                ),
            )
            for (subject_uid, canonical_predicate, object_uid), relationships
            in sorted(relationship_groups.items(), key=lambda item: tuple(map(str, item[0])))
        ]
        return SceneGraph(
            image_id=scene_graphs[0].image_id,
            source=_join_sources(*(graph.source for graph in scene_graphs)),
            objects=merged_objects,
            relationships=merged_relationships,
        )

    # ------------------------------------------------------------------
    # Object matching
    # ------------------------------------------------------------------

    def _match_objects(
        self,
        base_objects: list[SceneObject],
        incoming_objects: list[SceneObject],
    ) -> tuple[list[ObjectMatch], list[SceneObject]]:
        """Find best matches between base and incoming objects.

        Returns (matches, unmatched_incoming).  Uses a greedy best-first
        strategy: score every pair, then greedily pick the highest-scoring
        pair that hasn't been claimed yet.
        """
        candidates: list[ObjectMatch] = []

        for inc_obj in incoming_objects:
            for base_obj in base_objects:
                iou = self._compute_iou(base_obj, inc_obj)
                sem = self._semantic_similarity(base_obj, inc_obj)

                spatial_ok = (
                    not self.config.require_spatial_overlap
                    or iou >= self.config.iou_threshold
                    # Allow match when either side has no bbox
                    or base_obj.bbox is None
                    or inc_obj.bbox is None
                )
                semantic_ok = sem >= self.config.semantic_threshold

                if spatial_ok and semantic_ok:
                    candidates.append(ObjectMatch(base_obj, inc_obj, iou, sem))

        # greedy matching — highest combined score first
        candidates.sort(key=lambda m: m.iou + m.semantic_score, reverse=True)
        used_base: set[str] = set()
        used_incoming: set[str] = set()
        matches: list[ObjectMatch] = []

        for cand in candidates:
            if cand.obj_a.uid in used_base or cand.obj_b.uid in used_incoming:
                continue
            matches.append(cand)
            used_base.add(cand.obj_a.uid)
            used_incoming.add(cand.obj_b.uid)

        unmatched = [o for o in incoming_objects if o.uid not in used_incoming]
        return matches, unmatched

    # ------------------------------------------------------------------
    # Object merging
    # ------------------------------------------------------------------

    def _merge_objects(self, objects: list[SceneObject]) -> SceneObject:
        """Merge a matched object group, calculating aggregate fields once."""
        primary=max(objects, key=lambda obj: obj.confidence)
        return SceneObject(
            label=primary.label, # take the label from the most confident object
            bbox=self._merge_bboxes(objects),
            attributes=sorted({attribute for obj in objects for attribute in obj.attributes}),
            confidence=self._merge_confidences([obj.confidence for obj in objects]),
            source=_join_sources(*(obj.source for obj in objects)),
            canonical_label=max(objects, key=lambda obj: obj.confidence).canonical_label,
            uid=min((obj.uid for obj in objects), key=str),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(a: SceneObject, b: SceneObject) -> float:
        if a.bbox is None or b.bbox is None:
            return 0.0
        return a.bbox.iou(b.bbox)

    @staticmethod
    def _semantic_similarity(a: SceneObject, b: SceneObject) -> float:
        """Semantic similarity between two objects' canonical labels.

        Returns 1.0 for an exact label match, else falls back to Wu-Palmer.
        """
        la = a.canonical_label
        lb = b.canonical_label
        if la == lb:
            return 1.0
        return wup_confidence(la, lb)

    @staticmethod
    def _merge_bboxes(objects: list[SceneObject]) -> BoundingBox | None:
        """Return the average bounding box of every object that supplies one."""
        bboxes = [obj.bbox for obj in objects if obj.bbox is not None]
        if not bboxes:
            return None
        return BoundingBox(
            x_min=sum(bbox.x_min for bbox in bboxes) / len(bboxes),
            y_min=sum(bbox.y_min for bbox in bboxes) / len(bboxes),
            x_max=sum(bbox.x_max for bbox in bboxes) / len(bboxes),
            y_max=sum(bbox.y_max for bbox in bboxes) / len(bboxes),
        )
    def _merge_confidences(self, confs: list[float]) -> float:
        """Calculate the confidence of a merged object from its constituent objects.

        If `inverse_confidence_calculation` is True, the confidence is calculated as the inverse of the product of the inverse confidences of its constituent objects.
        e.g. 2 objects with confidence 90% would result in a merged confidence of 99% (1 - (1-0.9)*(1-0.9) = 0.99). If `inverse_confidence_calculation` is False, the confidence is simply the maximum confidence of the constituent objects.
        """
        if not confs:
            return 0.0
        if self.config.inverse_confidence_calculation:
            product_inverse_confidence = 1.0
            for conf in confs:
                product_inverse_confidence *= (1.0 - conf)
            return 1.0 - product_inverse_confidence
        else:
            return max(confs)

def _join_sources(*sources: str) -> str:
    """Combine source name strings, deduplicating."""
    parts: list[str] = []
    for s in sources:
        for part in s.split("+"):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return "+".join(sorted(parts))
