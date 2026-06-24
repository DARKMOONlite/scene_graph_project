"""_summary_

A temporal stabaliser for the scene graph fusion pipeline. It takes in a list of scene graphs 
and outputs a list of stabalised scene graphs. 
The goal is for it to do 4 things:
1. reduce noise or flickering relationships between frames
3. entity identification and tracking. 
4. maintain temporal consistency of attributes across frames.

"""
from __future__ import annotations


from copy import deepcopy
from uuid import UUID, uuid4

from scene_graph_project.scene_graph_fusion.pipeline.models import BoundingBox, SceneGraph,SceneObject,Relationship
from scene_graph_project.scene_graph_fusion.pipeline.network_flow import NetworkFlow, NodeID, TrackingConfig
from boxmot.trackers import OccluBoost
import numpy as np
class TemporalStabaliser:
    """A temporal stabaliser for the scene graph fusion pipeline. It takes in a list of scene graphs 
    and outputs a list of stabalised scene graphs. 
    The goal is for it to do 4 things:
    1. reduce noise or flickering relationships between frames
    3. entity identification and tracking. 
    4. maintain temporal consistency of attributes across frames.

    """

    def __init__(self):
        pass

    def stabalise(self, graphs: list[SceneGraph]) -> list[SceneGraph]:
        """Stabalise the given list of scene graphs and return the stabalised graphs."""
        # entity_tracked_graphs = self.entity_tracking(graphs)
        pass
    

    
    def entity_tracking(self, ordered_graphs: list[SceneGraph], max_skip: int = 2) -> list[SceneGraph]:
        """Track entities across frames and assign consistent IDs to the same entity."""
        if not ordered_graphs:
            return []

        tracked_graphs = deepcopy(ordered_graphs)
        print(
            "[entity_tracking] starting | "
            f"frames={len(tracked_graphs)}, "
            f"objects={sum(len(g.objects) for g in tracked_graphs)}, "
            f"relationships={sum(len(g.relationships) for g in tracked_graphs)}"
        )
        nf = NetworkFlow(
            config=TrackingConfig(
                birth_cost=1.2,
                death_cost=1.2,
                birth_time_bias=0.02,
                death_time_bias=0.02,
                skip_penalty=0.5,
                max_skip=2,
                max_candidates_per_node=5,
                min_track_length=2,
                viz_show_source_sink=False,
            )
        )
        _mcf, _node_ids, filtered_tracks = nf.m_cost_flow(tracked_graphs)
        aligned_scene_graphs = self.align_scene_graph_ids(filtered_tracks, tracked_graphs)
        nf.visualise(_mcf,_node_ids)


        return aligned_scene_graphs
    
    def mot_tracking(self, graphs: list[SceneGraph],images:list[np.ndarray]) -> list[SceneGraph]:
        """Use a multi-object tracking approach to track entities across frames."""
        tracker = OccluBoost()
        
        scene_pairs = zip(graphs,images)
        for scene in scene_pairs:
            detections = np.ndarray((len(scene[0].objects), 6)) # dets: (N, 6) array with [x1, y1, x2, y2, conf, cls] per detection
            for idx, obj in enumerate(scene[0].objects):
                # Assuming obj.bbox is in the format [x_center, y_center, width, height]
                
                if isinstance(obj.bbox, BoundingBox):
                    detections[idx] = [obj.bbox.x_min, obj.bbox.y_min, obj.bbox.x_max, obj.bbox.y_max, obj.confidence, obj.canonical_label]
                else:
                    print(f"Object {obj.uid} has invalid bbox format: {obj.bbox}")
                
            
            tracks = tracker.update(scene[1],detections) # tracks: (M, 8) array with [x1, y1, x2, y2, id, conf, cls, det_ind] per track
            print(tracks)
            
        


    

    def align_scene_graph_ids(self, tracks: list[list[NodeID]], graphs: list[SceneGraph]) -> list[SceneGraph]:
        """Assign a persistent UUID per track and remap relationship UIDs accordingly."""
        node_to_track_uid: dict[NodeID, UUID] = {}
        for track in tracks:
            track_uid = uuid4()
            for node_id in track:
                node_to_track_uid[node_id] = track_uid

        # Per-frame mapping from old object UID -> new tracked UID.
        frame_uid_remaps: list[dict[UUID, UUID]] = []
        for graph_idx, graph in enumerate(graphs):
            uid_remap: dict[UUID, UUID] = {}
            for obj_idx, obj in enumerate(graph.objects):
                node_id = NodeID(graph_idx, obj_idx)
                old_uid = obj.uid
                new_uid = node_to_track_uid.get(node_id, uuid4())
                obj.uid = new_uid
                uid_remap[old_uid] = new_uid
            frame_uid_remaps.append(uid_remap)

        for graph_idx, graph in enumerate(graphs):
            uid_remap = frame_uid_remaps[graph_idx]
            for rel in graph.relationships:
                if rel.subject_uid in uid_remap:
                    rel.subject_uid = uid_remap[rel.subject_uid]
                if rel.object_uid in uid_remap:
                    rel.object_uid = uid_remap[rel.object_uid]

        return graphs
    
    def reduce_noise(self, graphs: list[SceneGraph]) -> list[SceneGraph]:
        """Reduce noise or flickering relationships between frames."""
        return graphs
