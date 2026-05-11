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

    prefer_higher_confidence: bool = True
    """When merging, prefer the label with the higher confidence score."""
    
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

    def track(self,other:ObjectMatch) -> bool:
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
        if len(graphs) == 1:
            return graphs[0]

        # Start with the first graph as base and fold in the rest.
        merged = self._copy_graph(graphs[0])
        for graph in graphs[1:]:
            merged = self._merge_pair(merged, graph)
            
        if self.config.label_set is not None: # filter out objects whose canonical label is not in the allowed set
            merged.objects = [o for o in merged.objects if o.canonical_label in self.config.label_set]
            # also filter relationships to only include those whose subject and object are still present
            valid_uids = set(o.uid for o in merged.objects)
            merged.relationships = [r for r in merged.relationships if r.subject_uid in valid_uids and r.object_uid in valid_uids]    
        
        return merged

    # ------------------------------------------------------------------
    # Pairwise merge
    # ------------------------------------------------------------------
    
    def _merge_pair(self, base: SceneGraph, incoming: SceneGraph) -> SceneGraph:
        """Merge *incoming* into *base*, returning the updated graph."""
        matches, unmatched = self._match_objects(base.objects, incoming.objects)

        # uid of incoming object → uid of merged object
        uid_remap: dict[int, int] = {}

        # --- merge matched objects ---
        for match in matches:
            merged_obj = self._merge_objects(match.obj_a, match.obj_b)
            # replace the base object in-place
            for i, obj in enumerate(base.objects):
                if obj.uid == match.obj_a.uid:
                    base.objects[i] = merged_obj
                    break
            uid_remap[match.obj_b.uid] = merged_obj.uid

        # --- add unmatched incoming objects ---
        for obj in unmatched:
            base.objects.append(obj)
            uid_remap[obj.uid] = obj.uid  # identity mapping

        # --- merge relationships ---
        for rel in incoming.relationships:
            new_subj = uid_remap.get(rel.subject_uid, rel.subject_uid)
            new_obj = uid_remap.get(rel.object_uid, rel.object_uid)
            new_rel = Relationship(
                subject_uid=new_subj,
                predicate=rel.predicate,
                object_uid=new_obj,
                canonical_predicate=rel.canonical_predicate,
                confidence=rel.confidence,
                source=rel.source,
            )
            if not self._has_duplicate_relationship(base.relationships, new_rel):
                base.relationships.append(new_rel)

        return base

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

    def _merge_objects(self, a: SceneObject, b: SceneObject) -> SceneObject:
        """Merge two matched SceneObjects into one.

        Keeps the higher-confidence label (or the first if equal), unions
        attributes, and takes the tighter / higher-confidence bounding box.
        """
        if self.config.prefer_higher_confidence and b.confidence > a.confidence:
            primary, secondary = b, a
        else:
            primary, secondary = a, b

        merged_attrs = list(dict.fromkeys(a.attributes + b.attributes))
        merged_bbox = self._merge_bboxes(a, b)
        sources = _join_sources(a.source, b.source)

        return SceneObject(
            label=primary.label,
            bbox=merged_bbox,
            attributes=merged_attrs,
            confidence=max(a.confidence, b.confidence),
            source=sources,
            canonical_label=primary.canonical_label,
            uid=a.uid,  # keep the base UID for stable references
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
    def _merge_bboxes(a: SceneObject, b: SceneObject) -> BoundingBox | None:
        """Return the average bounding box when both exist; otherwise whichever exists."""
        if a.bbox is None:
            return b.bbox
        if b.bbox is None:
            return a.bbox
        return BoundingBox(
            x_min=(a.bbox.x_min + b.bbox.x_min) / 2,
            y_min=(a.bbox.y_min + b.bbox.y_min) / 2,
            x_max=(a.bbox.x_max + b.bbox.x_max) / 2,
            y_max=(a.bbox.y_max + b.bbox.y_max) / 2,
        )

    @staticmethod
    def _has_duplicate_relationship(
        existing: list[Relationship], candidate: Relationship
    ) -> bool:
        """Check if an equivalent relationship already exists."""
        for r in existing:
            if (
                r.subject_uid == candidate.subject_uid
                and r.canonical_predicate == candidate.canonical_predicate
                and r.object_uid == candidate.object_uid
            ):
                return True
        return False

    @staticmethod
    def _copy_graph(graph: SceneGraph) -> SceneGraph:
        """Shallow-copy a graph so the original isn't mutated."""
        return SceneGraph(
            image_id=graph.image_id,
            source=graph.source,
            objects=list(graph.objects),
            relationships=list(graph.relationships),
        )


def _join_sources(*sources: str) -> str:
    """Combine source name strings, deduplicating."""
    parts: list[str] = []
    for s in sources:
        for part in s.split("+"):
            part = part.strip()
            if part and part not in parts:
                parts.append(part)
    return "+".join(parts)
