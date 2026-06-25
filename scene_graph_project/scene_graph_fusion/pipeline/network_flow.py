from copy import deepcopy
from typing import NamedTuple

from ortools.graph.python import min_cost_flow
from dataclasses import dataclass
from scene_graph_project.scene_graph_fusion.pipeline.models import NodeID, SceneGraph, SceneGraphShape,SceneObject,Relationship
from collections import Counter
import matplotlib.pyplot as plt
@dataclass
class TrackingConfig:
    birth_cost: float = 1.2        # base cost for starting a new track at any detection
    death_cost: float = 1.2        # base cost for ending a track at any detection
    birth_time_bias: float = 0.2  # additive bias per frame index for birth arcs
    death_time_bias: float = 0.2  # additive bias per distance-to-end for death arcs
    skip_penalty: float = 0.5      # extra cost per skipped frame in a gap transition
    continuation_bonus: float = 0.35  # reward for continuing a track instead of terminating
    max_skip: int = 2              # maximum frames a track can bridge without a detection
    max_candidates_per_node: int = 5
    min_track_length: int = 2      # minimum detections required to keep a track
    transition_cost_cap: int = 180  # keep transition costs in a stable range after scaling
    cost_scale: int = 100
    bb_penalty_weight: float = 1.0
    label_penalty_weight: float = 0.5
    distance_penalty_weight: float = 0.25
    relationship_penalty_weight: float = 0.25
    viz_show_source_sink: bool = False   # whether to show source/sink arcs in visualisation
    viz_show_zero_flow: bool = True           # whether to show arcs with zero flow in visualisation



class SplitNodeMaps(NamedTuple):
    in_node_ids: dict[NodeID, int]
    out_node_ids: dict[NodeID, int]

# Cost for moving from S -> T = 10,000 
# Cost for S -> detection = birth_cost * (frame_idx + 1) * cost_scale
# Cost for detection -> T = death_cost * (remaining_frames) * cost_scale
# Cost for detection(t) -> detection(t2)  = 0 <= <=


class NetworkFlow:
    def __init__(self, config: TrackingConfig = TrackingConfig()):
        self.config = config
        self.last_tracks: list[list[NodeID]] = []
        self.last_filtered_tracks: list[list[NodeID]] = []
    

    def m_cost_flow(
        self, tracked_graphs: list[SceneGraph]
    ) -> tuple[min_cost_flow.SimpleMinCostFlow, SplitNodeMaps, list[list[NodeID]]]:
        """Build and solve a min-cost flow tracking graph.

        Uses a global Source (S) and Sink (T) so tracks can start or end at any
        frame.  Skip arcs (up to max_skip frames apart) bridge temporary
        disappearances without requiring a shared MISSING hub that loses identity.

        Supply model
        ------------
        S supplies total_detections units; T absorbs the same.
        A S -> T overflow arc absorbs any units not routed through real detections,
        keeping the problem always feasible regardless of frame sizes.
        """
        if len(tracked_graphs) < 2:
            raise ValueError("Need at least 2 frames for tracking")

        mcf = min_cost_flow.SimpleMinCostFlow()

        # ---- Node IDs: split each detection into in-node and out-node ----
        in_node_ids: dict[NodeID, int] = {}
        out_node_ids: dict[NodeID, int] = {}
        n = 0
        for t, graph in enumerate(tracked_graphs):
            for obj_id in range(len(graph.objects)):
                node_id = NodeID(t, obj_id)
                in_node_ids[node_id] = n
                n += 1
                out_node_ids[node_id] = n
                n += 1

        total_detections = len(in_node_ids)
        S = n;  n += 1   # global source
        T = n;  n += 1   # global sink

        birth_cost_scaled = self._scaled_int(self.config.birth_cost, min_value=1)
        death_cost_scaled = self._scaled_int(self.config.death_cost, min_value=1)
        birth_bias_scaled = self._scaled_int(self.config.birth_time_bias)
        death_bias_scaled = self._scaled_int(self.config.death_time_bias)
        skip_scale = self._scaled_int(self.config.skip_penalty)
        continuation_bonus_scaled = self._scaled_int(self.config.continuation_bonus)

        # ---- Arcs ----

        # 1. Overflow: S -> T absorbs unused supply so the problem is always feasible.
        # Keep this expensive to avoid swallowing flow that should go through tracks.
        """When a transition arc is used (e.g. out(d0) → in(d1)), it occupies in(d1)'s capacity. 
        But S still has N units to route — the unit that would have taken the S → in(d1) birth arc is now blocked (capacity=1 already used).
        That stranded unit has only one escape: the overflow arc at 50 × (birth + death) ≈ 240 scaled units."""
        overflow_cost = 50 * (birth_cost_scaled + death_cost_scaled)
        mcf.add_arc_with_capacity_and_unit_cost(S, T, total_detections, overflow_cost)

        # 2. Birth arcs: S -> in-node (a track can start at any detection in any frame).
        for node_id, in_node in in_node_ids.items():
            birth_cost = birth_cost_scaled + (birth_bias_scaled * node_id.frame_idx)
            mcf.add_arc_with_capacity_and_unit_cost(S, in_node, 1, birth_cost)

        # 3. Death arcs: out-node -> T (a track can end at any detection in any frame).
        last_frame_idx = len(tracked_graphs) - 1
        for node_id, out_node in out_node_ids.items():
            death_cost = death_cost_scaled + (death_bias_scaled * (last_frame_idx - node_id.frame_idx))
            mcf.add_arc_with_capacity_and_unit_cost(out_node, T, 1, death_cost)

        # 4. Intranode arcs: in-node -> out-node with capacity 1.
        for node_id in in_node_ids:
            mcf.add_arc_with_capacity_and_unit_cost(in_node_ids[node_id], out_node_ids[node_id], 1, 0)

        # 5. Transition arcs: out(t) -> in(t2) for t2 in [t+1, t+max_skip].
        #    Each extra skipped frame adds skip_penalty so direct matches are preferred
        #    and identity is never lost through a shared hub.
        for t in range(len(tracked_graphs)):
            frame_t = [
                (node_id.obj_idx, out_node, tracked_graphs[t].objects[node_id.obj_idx])
                for node_id, out_node in out_node_ids.items()
                if node_id.frame_idx == t
            ]
            for t2 in range(t + 1, min(t + self.config.max_skip + 1, len(tracked_graphs))):
                skip_extra = skip_scale * (t2 - t - 1)
                frame_t2 = [
                    (node_id2.obj_idx, in_node2, tracked_graphs[t2].objects[node_id2.obj_idx])
                    for node_id2, in_node2 in in_node_ids.items()
                    if node_id2.frame_idx == t2
                ]
                for obj_id1, out_node1, obj1 in frame_t:
                    rel_a = [
                        r for r in tracked_graphs[t].relationships
                        if r.subject_uid == obj1.uid or r.object_uid == obj1.uid
                    ]
                    candidates: list[tuple[int, int]] = []
                    for obj_id2, in_node2, obj2 in frame_t2:
                        rel_b = [
                            r for r in tracked_graphs[t2].relationships
                            if r.subject_uid == obj2.uid or r.object_uid == obj2.uid
                        ]
                        base = self.entity_cost(obj1, obj2, rel_a, rel_b, len(tracked_graphs))
                        transition_cost = max(1, base + skip_extra - continuation_bonus_scaled)
                        transition_cost = min(self.config.transition_cost_cap, transition_cost)
                        candidates.append((transition_cost, in_node2))

                    candidates.sort(key=lambda x: x[0]) # only add the best few candidates to limit the graph size and focus on the most promising matches
                    for transition_cost, in_node2 in candidates[: self.config.max_candidates_per_node]:
                        mcf.add_arc_with_capacity_and_unit_cost(out_node1, in_node2, 1, transition_cost)

        # ---- Supplies ----
        # S injects one unit per detection; the overflow arc absorbs whatever is not
        # routed through real birth arcs, keeping the problem balanced and feasible.
        supplies = [0] * n
        # for node_id, node in in_node_ids.items():
        #     if node_id.frame_idx == 0:
        #         supplies[node] += 1  # only detections in the first frame are guaranteed to be births; later ones could be skipped by a track starting earlier
        #     if node_id.frame_idx == len(tracked_graphs) - 1:
        #         supplies[node] -= 1  # only detections in the last frame are guaranteed to be deaths; earlier ones could be skipped by a track ending later
        supplies[S] = total_detections
        supplies[T] = -(total_detections)
        if sum(supplies) != 0:
            raise ValueError(f"Supplies must sum to zero {sum(supplies)}")
        mcf.set_nodes_supplies(list(range(n)), supplies)
        

        status = mcf.solve()
        if status != mcf.OPTIMAL:
            print(f"MCF solver failed with status {status}")
            exit(1)

        tracks = self.extract_tracks(mcf, in_node_ids, out_node_ids, S, T)
        filtered_tracks = self.filter_tracks(tracks)
        self.last_tracks = tracks
        self.last_filtered_tracks = filtered_tracks

        print(
            f"Extracted {len(tracks)} tracks "
            f"({len(filtered_tracks)} kept with >= {self.config.min_track_length} detections)."
        )
        for track_idx, track in enumerate(filtered_tracks, start=1):
            print(f"  Track {track_idx}:", end="")
            for frame_idx, obj_idx in track:
                print(f" (Frame {frame_idx}: Obj {obj_idx})", end="")
            print("")

        return mcf, SplitNodeMaps(in_node_ids=in_node_ids, out_node_ids=out_node_ids), filtered_tracks
  

    def m_cost_circular_flow(
                self, tracked_graphs: list[SceneGraph]
    ) -> tuple[min_cost_flow.SimpleMinCostFlow, SplitNodeMaps, list[list[NodeID]]]:
        """Build a circular flow system for determing object tracks. 
        in this there exists a single node that connects the start of the first scene graph to the end of the last, creating a loop
        flow should try to connect to a node in the next frame, 

        Args:
            tracked_graphs (list[SceneGraph]): _description_

        Returns:
            tuple[min_cost_flow.SimpleMinCostFlow, SplitNodeMaps, list[list[NodeID]]]: _description_
        """
        if len(tracked_graphs) < 2:
            raise ValueError("Need at least 2 frames for tracking")

        mcf = min_cost_flow.SimpleMinCostFlow()
        in_node_ids: dict[NodeID, int] = {}
        out_node_ids: dict[NodeID, int] = {}
        n = 0
        for t, graph in enumerate(tracked_graphs):
            for obj_id in range(len(graph.objects)):
                node_id = NodeID(t, obj_id)
                in_node_ids[node_id] = n
                n += 1
                out_node_ids[node_id] = n
                n += 1

        total_detections = len(in_node_ids)
        S = n;  n += 1   # global source
        T = n;  n += 1   # global sink
        
        birth_cost_scaled = self._scaled_int(self.config.birth_cost, min_value=1)
        death_cost_scaled = self._scaled_int(self.config.death_cost, min_value=1)
        birth_bias_scaled = self._scaled_int(self.config.birth_time_bias)
        death_bias_scaled = self._scaled_int(self.config.death_time_bias)
        skip_scale = self._scaled_int(self.config.skip_penalty)
        continuation_bonus_scaled = self._scaled_int(self.config.continuation_bonus)
        
        # 1. Overflow, no overflow arc, as it causes large costs to be attributed to movements through the graph 
        
        # 2. Intranode arcs: in-node -> out-node with capacity 1.
        for node_id in in_node_ids:
            mcf.add_arc_with_capacity_and_unit_cost(in_node_ids[node_id], out_node_ids[node_id], 1, 0)
        # 3. Transition arcs: out(t) -> in(t2) for t2 in [t+1, t+max_skip].
        for t in range(len(tracked_graphs)):
            frame_t = [
                (node_id.obj_idx, out_node, tracked_graphs[t].objects[node_id.obj_idx])
                for node_id, out_node in out_node_ids.items()
                if node_id.frame_idx == t
            ]
            for t2 in range(t + 1, min(t + self.config.max_skip + 1, len(tracked_graphs))):
                skip_extra = skip_scale * (t2 - t - 1)
                frame_t2 = [
                    (node_id2.obj_idx, in_node2, tracked_graphs[t2].objects[node_id2.obj_idx])
                    for node_id2, in_node2 in in_node_ids.items()
                    if node_id2.frame_idx == t2
                ]
                for obj_id1, out_node1, obj1 in frame_t:
                    rel_a = [
                        r for r in tracked_graphs[t].relationships
                        if r.subject_uid == obj1.uid or r.object_uid == obj1.uid
                    ]
                    candidates: list[tuple[int, int]] = []
                    for obj_id2, in_node2, obj2 in frame_t2:
                        rel_b = [
                            r for r in tracked_graphs[t2].relationships
                            if r.subject_uid == obj2.uid or r.object_uid == obj2.uid
                        ]
                        base = self.entity_cost(obj1, obj2, rel_a, rel_b, len(tracked_graphs))
                        transition_cost = max(1, base + skip_extra - continuation_bonus_scaled)
                        transition_cost = min(self.config.transition_cost_cap, transition_cost)
                        candidates.append((transition_cost, in_node2))

                    candidates.sort(key=lambda x: x[0]) # only add the best few candidates to limit the graph size and focus on the most promising matches
                    for transition_cost, in_node2 in candidates[: self.config.max_candidates_per_node]:
                        mcf.add_arc_with_capacity_and_unit_cost(out_node1, in_node2, 1, transition_cost)
        # 4. Birth arcs: S -> in-node (a track can start at any detection in any frame).
        for node_id, in_node in in_node_ids.items():
            birth_cost = birth_cost_scaled + (birth_bias_scaled * node_id.frame_idx)
            mcf.add_arc_with_capacity_and_unit_cost(S, in_node, 1, birth_cost)
            print(f"Added birth arc: S -> in({node_id}) with cost {birth_cost}")

        # 5. Death arcs: out-node -> T (a track can end at any detection in any frame).
        last_frame_idx = len(tracked_graphs) - 1
        for node_id, out_node in out_node_ids.items():
            death_cost = death_cost_scaled + (death_bias_scaled * (last_frame_idx - node_id.frame_idx))
            mcf.add_arc_with_capacity_and_unit_cost(out_node, T, 1, death_cost)
            print(f"Added death arc: out({node_id}) -> T with cost {death_cost}")
        #6. Circular arc: T->S with zero cost and capacity equal to total detections, allowing flow to loop back from the end to the start
        mcf.add_arc_with_capacity_and_unit_cost(T, S, total_detections, 0)
        
        
        # ----- Supplies ----
        supplies = [0] * n
        for node_id, node in in_node_ids.items():
            supplies[node] -= 1  # each detection must be entered once
        for node_id, node in out_node_ids.items():
            supplies[node] += 1  # each detection must be exited once
        # S and T have zero net supply; they are just transit nodes in the circular flow
        if sum(supplies) != 0:
            raise ValueError(f"Supplies must sum to zero {sum(supplies)}")
        mcf.set_nodes_supplies(list(range(n)), supplies)
        
        status = mcf.solve()
        if status != mcf.OPTIMAL:
            print(f"MCF solver failed with status {status}")
            exit(1)
        tracks = self.extract_tracks(mcf, in_node_ids, out_node_ids, S, T)
        filtered_tracks = self.filter_tracks(tracks)
        self.last_tracks = tracks
        self.last_filtered_tracks = filtered_tracks
        print(
            f"Extracted {len(tracks)} tracks "
            f"({len(filtered_tracks)} kept with >= {self.config.min_track_length} detections)."
        )
        for track_idx, track in enumerate(filtered_tracks, start=1):
            print(f"  Track {track_idx}:", end="")
            for frame_idx, obj_idx in track:
                print(f" (Frame {frame_idx}: Obj {obj_idx})", end="")
            print("")
        return mcf, SplitNodeMaps(in_node_ids=in_node_ids, out_node_ids=out_node_ids), filtered_tracks
    
    
    def _scaled_int(self, value: float, min_value: int = 0) -> int:
        """Convert a float cost to an integer cost for OR-Tools min-cost flow."""
        return max(min_value, int(round(value * self.config.cost_scale)))
    
    def extract_tracks(
        self,
        mcf: min_cost_flow.SimpleMinCostFlow,
        in_node_ids: dict[NodeID, int],
        out_node_ids: dict[NodeID, int],
        S: int,
        T: int,
    ) -> list[list[NodeID]]:
        """Extract individual tracks from the solved flow by following arcs from S.

        Each positive-flow arc from S to a real detection starts one track.
        The track is extended by following the unique positive-flow outgoing
        arc from each detection until no non-T successor exists.
        Skip arcs mean consecutive entries in a path may not be adjacent frames;
        the frame index in each tuple records the actual frame, so gaps are visible.
        """
        in_id_to_key = {nid: key for key, nid in in_node_ids.items()}
        out_id_to_key = {nid: key for key, nid in out_node_ids.items()}

        # Build positive-flow adjacency (arcs with flow > 0).
        outgoing: dict[int, list[int]] = {}
        for arc in range(mcf.num_arcs()):
            if mcf.flow(arc) <= 0:
                continue
            u, v = mcf.tail(arc), mcf.head(arc)
            outgoing.setdefault(u, []).append(v)
            
        tracks: list[list[NodeID]] = []
        # Every positive-flow arc from S that leads to an in-node (not T) is a track start.
        for start_in in outgoing.get(S, []):
            if start_in == T:  # skip the S->T overflow arc
                continue
            if start_in not in in_id_to_key:
                continue
            path = [in_id_to_key[start_in]]
            cur_in = start_in
            while True:
                # in -> out via capacity limiter
                cur_out = next((v for v in outgoing.get(cur_in, []) if v in out_id_to_key), None)
                if cur_out is None:
                    break
                # out -> next in via transition, ignoring out -> T death arcs.
                next_in = next((v for v in outgoing.get(cur_out, []) if v in in_id_to_key), None)
                if next_in is None:
                    break
                path.append(in_id_to_key[next_in])
                cur_in = next_in
            tracks.append(path)
        return tracks

    def filter_tracks(
        self, tracks: list[list[tuple[int, int]]]
    ) -> list[list[tuple[int, int]]]:
        """Discard tracks with fewer than min_track_length real detections."""
        return [track for track in tracks if len(track) >= self.config.min_track_length]
        
        
        
    def entity_cost(self,
                    obj1: SceneObject, 
                    obj2: SceneObject, 
                    rel_a:list[Relationship], 
                    rel_b:list[Relationship],
                    num_graphs:int,
                    min_value:int=0,
                    max_value:int=50,
                    ) -> int:
        """Calculate the cost of matching two entities based on their attributes and relationships. returns a value
        between 0 and max_value, where 0 means a perfect match and max_value means a very poor match."""
        bbox_penalty = self._bbox_penalty(obj1, obj2)
        label_penalty = self._label_penalty(obj1, obj2)
        relationship_penalty = self._relationship_penalty(obj1, obj2, rel_a, rel_b)
        distance_penalty = self._centroid_distance_penalty(obj1, obj2)

        cost = ( bbox_penalty +  label_penalty + relationship_penalty + distance_penalty)/4 * max_value
        # cost /= num_graphs # normalize by number of graphs to keep costs comparable across different track lengths

        return int(max(min_value, cost))

    @staticmethod
    def _bbox_penalty(obj1: SceneObject, obj2: SceneObject) -> float:
        if obj1.bbox is None and obj2.bbox is None: # if neither has a bbox, we can't penalise based on spatial info, so return a moderate penalty
            return 0.35
        if obj1.bbox is None or obj2.bbox is None: # if only one has a bbox, return a higher penalty
            return 0.7
        return 1.0 - obj1.bbox.iou(obj2.bbox)
    @staticmethod
    def _label_penalty(obj1: SceneObject, obj2: SceneObject) -> float:
        label_penalty = 0.0 if obj1.label == obj2.label else 1.0
        canonical_label_penalty = 0.0 if obj1.canonical_label == obj2.canonical_label else 1.0
        return label_penalty + canonical_label_penalty

    @staticmethod
    def _relationship_signature(obj: SceneObject, rels: list[Relationship]) -> Counter[tuple[str, str]]:
        signature: Counter[tuple[str, str]] = Counter()
        for rel in rels:
            role = "subject" if rel.subject_uid == obj.uid else "object"
            predicate = rel.canonical_predicate or rel.predicate
            signature[(role, predicate)] += 1
        return signature

    def _relationship_penalty(
        self,
        obj1: SceneObject,
        obj2: SceneObject,
        rel_a: list[Relationship],
        rel_b: list[Relationship],
    ) -> float:
        sig_a = self._relationship_signature(obj1, rel_a)
        sig_b = self._relationship_signature(obj2, rel_b)

        if not sig_a and not sig_b:
            return 0.0

        keys = set(sig_a) | set(sig_b)
        difference = sum(abs(sig_a[key] - sig_b[key]) for key in keys)
        total = sum(max(sig_a[key], sig_b[key]) for key in keys)
        return difference / total if total else 0.0
    
    def _centroid_distance_penalty(self, obj1: SceneObject, obj2: SceneObject) -> float:
        """ Calculate a penalty based on the distance between the centroids of the bounding boxes of two objects.
            If either object lacks a bounding box, return a default moderate penalty."""
        if obj1.bbox is None or obj2.bbox is None:
            return 0.5
        center1 = obj1.bbox.centre
        center2 = obj2.bbox.centre
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        max_distance = ((obj1.bbox.width / 2) ** 2 + (obj1.bbox.height / 2) ** 2) ** 0.5 # relative to the size of the bounding box
        return min(distance / max_distance, 1.0)


    def visualise(self, mcf: min_cost_flow.SimpleMinCostFlow, node_maps: SplitNodeMaps,show_source_sink:bool=True,shape:SceneGraphShape=SceneGraphShape.SPRING):
        """ Visualise the tracking graph with tracks highlighted.  """
        
        G = SceneGraph("")
        

        mcf:min_cost_flow.SimpleMinCostFlow = mcf
        for node_id, node in node_maps.in_node_ids.items():
            G.add_object(SceneObject(uid=int(node), label=f"In {node_id.obj_idx}"))
        for node_id, node in node_maps.out_node_ids.items():
            G.add_object(SceneObject(uid=int(node), label=f"Out {node_id.obj_idx}"))
        G.add_object(SceneObject(uid=int(mcf.num_nodes()-2), label="Source (S)"))
        G.add_object(SceneObject(uid=int(mcf.num_nodes()-1), label="Sink (T)"))
        
        
        G2 = deepcopy(G)
        
        for arc in range(mcf.num_arcs()):
            if not self.config.viz_show_zero_flow and mcf.flow(arc) <= 0:
                continue
            u, v = mcf.tail(arc), mcf.head(arc)
            
            cost = mcf.unit_cost(arc)
            flow = mcf.flow(arc)
            if (u == mcf.num_nodes()-2 or v == mcf.num_nodes()-1): # S or T
                if not self.config.viz_show_source_sink:
                    continue
                G.add_relationship(Relationship(subject_uid=int(u), object_uid=int(v), predicate=f"S->T cost={cost} flow={flow}"))
                G2.add_relationship(Relationship(subject_uid=int(u), object_uid=int(v), predicate=f"S->T cost={cost} flow={flow}"))
            else:
                G.add_relationship(Relationship(subject_uid=int(u), object_uid=int(v), predicate=str(cost)))
                G2.add_relationship(Relationship(subject_uid=int(u), object_uid=int(v), predicate=str(flow)))
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        subsets:dict[int,tuple[NodeID,bool]] = {value: (key, True) for key, value in node_maps.in_node_ids.items()} | \
        {value: (key, False) for key, value in node_maps.out_node_ids.items()} # id : (Node_id, is_in_node)
        subsets[int(mcf.num_nodes()-2)] = (NodeID(-1, -1), False) # S
        subsets[int(mcf.num_nodes()-1)] = (NodeID(100, -1), False) # T
        
        G.visualise(node_labels=True, 
                    edge_labels=True, 
                    shape=shape,
                    subsets=subsets,
                    axis=ax1,
                    show_plot=False)
        G2.visualise(node_labels=True, 
                    edge_labels=True, 
                    shape=shape,
                    subsets=subsets,
                    axis=ax2,
                    show_plot=False)
        
        ax1.set_title("Tracking Graph with Arc Costs")
        ax2.set_title("Tracking Graph with Arc Flows")
        plt.show()