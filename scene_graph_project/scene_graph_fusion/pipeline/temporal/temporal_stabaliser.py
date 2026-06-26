"""_summary_

A temporal stabaliser for the scene graph fusion pipeline. It takes in a list of scene graphs 
and outputs a list of stabalised scene graphs. 
The goal is for it to do 4 things:
1. reduce noise or flickering relationships between frames
3. entity identification and tracking. 
4. maintain temporal consistency of attributes across frames.

"""
from __future__ import annotations


from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID, uuid4

import cv2

from scene_graph_project.scene_graph_fusion.pipeline.models import BoundingBox, SceneGraph, SceneObject, Relationship
from scene_graph_project.scene_graph_fusion.pipeline.network_flow import NetworkFlow, NodeID, TrackingConfig
from scene_graph_project.scene_graph_fusion.pipeline.temporal.temporal_scene_graph import TemporalSceneGraph
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.bbox.occluboost.occluboost import OccluBoost
from boxmot.trackers.tracker_zoo import create_tracker
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

    def stabalise(self, graphs: list[SceneGraph]) -> TemporalSceneGraph:
        """Stabalise the given list of scene graphs and return the stabalised graphs."""
        # entity_tracked_graphs = self.entity_tracking(graphs)
        return TemporalSceneGraph(graphs=graphs)
    

    
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
    
    def mot_tracking(self, graphs: list[SceneGraph], images: list[np.ndarray], visualise: bool = False) -> TemporalSceneGraph:
        """Use a multi-object tracking approach to track entities across frames."""
        if not graphs:
            return TemporalSceneGraph(graphs=[])
        if len(graphs) != len(images):
            raise ValueError(f"Expected the same number of graphs and images, got {len(graphs)} and {len(images)}.")

        tracker: BaseTracker = create_tracker("botsort", reid_weights="clip_vehicleid.pt")
        # tracker: OccluBoost = OccluBoost(
        #     reid_model="ss.pt",
        #     recovery_appearance_thresh=0.4,
        #     ams_alpha0=0.5,
        #     ams_threshold=0.6,
        #     ams_buffer_size=10,
        #     ams_shrink_ratio=0.5,
            
        #     )  # type: ignore
        #  update parameters
        
        tracker.det_thresh
        
        
        tracked_graphs = deepcopy(graphs)
        label_to_class_id: dict[str, int] = {}
        track_to_nodes: dict[int, list[NodeID]] = defaultdict(list)
        tracker_images: list[np.ndarray] = []
        for graph_idx, (scene, image) in enumerate(zip(tracked_graphs, images)):
            detections = np.zeros((len(scene.objects), 6), dtype=np.float32)
            for idx, obj in enumerate(scene.objects):
                if isinstance(obj.bbox, BoundingBox):
                    class_id = label_to_class_id.setdefault(obj.canonical_label, len(label_to_class_id))
                    detections[idx] = [
                        obj.bbox.x_min,
                        obj.bbox.y_min,
                        obj.bbox.x_max,
                        obj.bbox.y_max,
                        obj.confidence,
                        class_id,
                    ]
                else:
                    detections[idx, 4] = obj.confidence

            tracks = tracker.update(detections, image)
            tracker_images.append(tracker.plot_results(image,show_trajectories=True) )
            for track in tracks:
                det_idx = int(track[7])
                if det_idx < 0 or det_idx >= len(scene.objects):
                    continue
                track_to_nodes[int(track[4])].append(NodeID(graph_idx, det_idx))
                    
        if visualise:
            for image in tracker_images:
                cv2.imshow("BoxMot",image)
                cv2.waitKey(0)
            # self._mot_visualise(tracked_graphs, images, track_to_nodes)
        
        temporal_graph = TemporalSceneGraph(graphs=tracked_graphs)
        for nodes in track_to_nodes.values():
            unique_nodes: list[NodeID] = list(dict.fromkeys(nodes))
            if len(unique_nodes) < 2:
                continue
            instances = [tracked_graphs[node.frame_idx].objects[node.obj_idx] for node in unique_nodes]
            scene_graphs = [tracked_graphs[node.frame_idx] for node in unique_nodes]
            temporal_graph.add_link(instances=instances, scene_graphs=scene_graphs, class_=instances[0].canonical_label)

        return temporal_graph
    # def _mot_visualise(self, graphs: list[SceneGraph], images: list[np.ndarray], track_to_nodes: dict[int, list[NodeID]]):
    #     """Visualise the tracking results by drawing bounding boxes and track IDs on the images."""
    #     images_with_tracks = []
    #     for graph_idx, (scene, image) in enumerate(zip(graphs, images)):
    #         for track_id, node_ids in track_to_nodes.items():
    #             for node_id in node_ids:
    #                 if node_id.frame_idx == graph_idx:
    #                     obj = scene.objects[node_id.obj_idx]
    #                     if isinstance(obj.bbox, BoundingBox):
    #                         cv2.rectangle(
    #                             image,
    #                             (int(obj.bbox.x_min), int(obj.bbox.y_min)),
    #                             (int(obj.bbox.x_max), int(obj.bbox.y_max)),
    #                             (0, 255, 0),
    #                             2,
    #                         )
    #                         cv2.putText(
    #                             image,
    #                             f"ID: {track_id}",
    #                             (int(obj.bbox.x_min), int(obj.bbox.y_min) - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX,
    #                             0.5,
    #                             (0, 255, 0),
    #                             2,
    #                         )
    #         images_with_tracks.append(image)
    #     for img in images_with_tracks:
    #         cv2.imshow("Tracking", img)
    #         cv2.waitKey(1)
    
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

