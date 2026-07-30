from __future__ import annotations
import enum
try:
	from nuscenes.nuscenes import NuScenes
except ImportError as exc:  # pragma: no cover - import-time guard
	raise ImportError(
		"nuscenes-devkit is required. Install it with 'pip install nuscenes-devkit'."
	) from exc

from dataclasses import dataclass
from typing import Any,TypedDict

import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scene_graph_project.scene_graph_fusion.pipeline import load_scene_graph_json,SceneGraph,BoundingBox,SceneObject


class CAMERA_CHANNELS(enum.Enum):
    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"
    
    @classmethod 
    def filter(cls, string:str)->str:
        for channel in reversed(cls): # check in reverse order to avoid partial matches (e.g. CAM_FRONT should be checked after CAM_FRONT_LEFT and CAM_FRONT_RIGHT)
            if channel.value in string:
                return channel.value
        raise ValueError(f"Invalid camera channel: {string}")


# CAMERA_CHANNELS = (
#     "CAM_FRONT",
#     "CAM_FRONT_LEFT",
#     "CAM_FRONT_RIGHT",
#     "CAM_BACK",
#     "CAM_BACK_LEFT",
#     "CAM_BACK_RIGHT",
# )



class Camera2dResults(TypedDict):
    CAM_FRONT: BoundingBox | None
    CAM_FRONT_LEFT: BoundingBox | None
    CAM_FRONT_RIGHT: BoundingBox | None
    CAM_BACK: BoundingBox | None
    CAM_BACK_LEFT: BoundingBox | None
    CAM_BACK_RIGHT: BoundingBox | None
    

@dataclass
class Annotation2d:
    token:str
    type:str
    results:Camera2dResults|BoundingBox # mapping from annotation token to its 2D bounding box in this camera, or None if it is not visible in this camera

    def get_bb(self, camera_channel:str) ->BoundingBox:
        if isinstance(self.results,Camera2dResults):
            return self.results[camera_channel]
        elif isinstance(self.results,BoundingBox):
            return self.results
        else:
            raise ValueError(f"Unexpected type for results: {type(self.results)}")
    def get_single_camera(self,camera_channel:str) -> Annotation2d:
        if isinstance(self.results,dict):
            return Annotation2d(token=self.token, type=self.type, results=self.results[camera_channel])
        elif isinstance(self.results,BoundingBox):
            return Annotation2d(token=self.token, type=self.type, results=self.results)
        else:
            raise ValueError(f"Unexpected type for results: {type(self.results)}")

def to_transform(translation:tuple[float,float,float], rotation:tuple[float,float,float,float]) -> RigidTransform:
    R = Rotation.from_quat(rotation,scalar_first=True).as_matrix()
    T = np.eye(4)
    T[  :3, :3] = R
    T[:3,  3] = translation
    return RigidTransform(T)


def project_to_2d(camera_pose:RigidTransform, camera_intrinsics:np.ndarray, point:RigidTransform)->tuple[float,float]|None:
    point_in_camera_frame = camera_pose.inv() * point
    x, y, z = point_in_camera_frame.translation
    if z <= 0:
        return None  # Point is behind the camera, cannot be projected
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    
    u = fx*(x/z) + cx
    v = fy*(y/z) + cy
    
    return u, v


def within_image_bounds(u:float, v:float, image_width:int, image_height:int) -> bool:
    return 0 <= u < image_width and 0 <= v < image_height

def corner_to_bounding_box(corners:list[tuple[float,float]],image_bounds:tuple[int,int]) -> tuple[float,float,float,float]|None:
    if not corners or len(corners) == 0:
        return None
    u_min = min(corner[0] for corner in corners)
    v_min = min(corner[1] for corner in corners)
    u_max = max(corner[0] for corner in corners)
    v_max = max(corner[1] for corner in corners)
    # u_min = max(0, u_min)
    # v_min = max(0, v_min)
    # u_max = min(image_bounds[0], u_max)
    # v_max = min(image_bounds[1], v_max)
    return u_min, v_min, u_max, v_max


def project_annotations_to_2d(sample:Sample,image_resolution:tuple[int,int]) -> list[Annotation2d]:
    """for each annotation in the sample, project its 3D bounding box to each camera and compute the 2D bounding box in the image plane. If the annotation is not visible in a camera, the result for that camera will be None.

    Args:
        sample (Sample): the sample for which to compute the bounding boxes
        image_resolution (tuple[int,int]): _description_

    Returns:
        list[Annotation2d]: _description_
    """
    results:list[Annotation2d] = []
    for annotation in tqdm(sample.annotations):
        corners_3d:list[tuple[float,float,float]] = annotation.get_bounding_box_corners()
        
        camera_results:Camera2dResults = {channel.value: None for channel in CAMERA_CHANNELS}
        for camera in sample.cameras.values():
            corners2d:list[tuple[float,float]] = []
            for corner in corners_3d:
                pixels = project_to_2d(camera.transformation, camera.camera_intrinsics, to_transform(corner, (1,0,0,0)))
                if pixels is not None and within_image_bounds(pixels[0], pixels[1], image_resolution[0], image_resolution[1]):
                    corners2d.append(pixels)
            bounding_box = corner_to_bounding_box(corners2d, image_resolution)
            if bounding_box is None:
                camera_results[camera.channel] = None
            else:
                camera_results[camera.channel] =BoundingBox( 
                    x_min= bounding_box[0],
                    y_min= bounding_box[1],
                    x_max= bounding_box[2],
                    y_max= bounding_box[3],
                )
        results.append(Annotation2d(token=annotation.token, results=camera_results,type=annotation.category_name))    
    return results


def organize_bounding_boxes_by_camera(annotations:list[Annotation2d],image_resolution:tuple[int,int]) -> dict[str,list[Annotation2d]]:
    results:dict[str,list[Annotation2d]] = {channel.value: [] for channel in CAMERA_CHANNELS}
    for annotation in annotations:
        for camera_channel, bounding_box in annotation.results.items():
            if bounding_box is not None:
                results[camera_channel].append(annotation.get_single_camera(camera_channel))
    return results

def anno2d_to_scene_object(anno:Annotation2d) -> SceneObject:
    if isinstance(anno.results,dict):
        raise ValueError("Expected a single bounding box, but got Camera2dResults. This function should only be used for annotations that have been processed for a single camera.")


    return SceneObject(
        label= anno.type,
        bbox=anno.results,
        uid=anno.token,
    )


def annotate_image(image_path: str, annotations: list[BoundingBox], dataroot: str) -> Image.Image:
    """Load image from dataroot/image_path and draw bounding boxes on it."""
    img = Image.open(f"{dataroot}/{image_path}").convert("RGWindow, TimeSegment,B")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.imshow(img)
    ax.axis("off")
    for box in annotations:
        rect = mpatches.Rectangle(
            (box["u_min"], box["v_min"]),
            box["u_max"] - box["u_min"],
            box["v_max"] - box["v_min"],
            linewidth=1, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
    fig.canvas.draw()
    annotated = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return annotated


# Layout: front cameras on top row, back cameras on bottom row
_CAMERA_LAYOUT = [
    (0, 0, "CAM_FRONT_LEFT"),
    (0, 1, "CAM_FRONT"),
    (0, 2, "CAM_FRONT_RIGHT"),
    (1, 0, "CAM_BACK_LEFT"),
    (1, 1, "CAM_BACK"),
    (1, 2, "CAM_BACK_RIGHT"),
]


def visualise_results(sample: "Sample", results: list[Annotation2d], dataroot: str) -> None:
    """Show all 6 camera views with projected bounding boxes in a single figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"Sample {sample.timestamp}")

    for row, col, channel in _CAMERA_LAYOUT:
        boxes = [ann.results[channel] for ann in results if ann.results[channel] is not None]
        labels = [ann.type for ann in results if ann.results[channel] is not None]
        img = Image.open(f"{dataroot}/{sample.cameras[channel].filename}").convert("RGB")
        ax = axes[row][col]
        ax.imshow(img)
        ax.set_title(channel, fontsize=8)
        ax.axis("off")
        for box, label in zip(boxes, labels):
            rect = mpatches.Rectangle(
                (box.x_min, box.y_min),
                box.x_max - box.x_min,
                box.y_max - box.y_min,
                linewidth=1, edgecolor="red", facecolor="none"
            )
            ax.text(box.x_min, box.y_min, label, color="red", fontsize=6, verticalalignment="top")
            
            ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


@dataclass
class Annotation:
    """
    Args:
    
        category_name: name of the category
        token: token identifier
        instance_token: token identifier for the instance
        visibility: visibility of the annotation
        transformation: the annotation's pose in the global frame at the time of the sample
        
    """
    category_name:str
    token:str
    instance_token:str
    visibility:float
    transformation:RigidTransform
    size:tuple[float,float,float]
    next:str
    prev:str
    
    def get_bounding_box_corners(self) -> list[tuple[float,float,float]]:
        # NuScenes size is [width, length, height]; box is centred on self.transformation.translation
        w, l, h = self.size
        corners = [
            ( l/2,  w/2, -h/2),
            ( l/2, -w/2, -h/2),
            (-l/2, -w/2, -h/2),
            (-l/2,  w/2, -h/2),
            ( l/2,  w/2,  h/2),
            ( l/2, -w/2,  h/2),
            (-l/2, -w/2,  h/2),
            (-l/2,  w/2,  h/2),
        ]
        return [self.transformation.apply(np.array(corner)) for corner in corners]
    
@dataclass
class CameraSample:
    """
    Args:
        image_token: the unique sample_data token for this individual camera frame.
        sample_token: the scene-level sample token shared across all 6 cameras at
            the same timestep (``sample_data["sample_token"]``).  Used to
            correlate frames across channels and look up aligned timestamps.
        scene_token: token for the scene this frame belongs to.
        channel: CAM_FRONT, CAM_FRONT_LEFT, etc.
        filename: relative path to the image file.
        prev: sample_data token of the previous frame in this channel's chain.
        next: sample_data token of the next frame in this channel's chain.
        transformation: the camera's pose in the global frame at the time of the sample.
        is_key_frame: whether this frame is a key frame (annotated sample).
    """
    image_token: str
    sample_token: str
    scene_token:str
    channel:str
    filename:str
    prev:str
    next:str
    timestamp:int
    transformation:RigidTransform # the camera's pose in the global frame at the time of the sample
    is_key_frame:bool = False
    camera_intrinsics:np.ndarray|None = None
    image_type:str|None = None # e.g. "sample" or "sweep"
    next_sample: str | None = None  # image_token of the next key frame in this channel; None for sweeps
    prev_sample: str | None = None  # image_token of the previous key frame in this channel; None for sweeps



@dataclass
class Sample():
    """A sample is a set of camera photos from a specific timestamp, including the annotations visable in those images.

    Returns:
        _type_: _description_
    """
    #TODO use this instead of get_camera_sample_lists
    timestamp:int
    annotations:list[Annotation]
    cameras:dict[str,CameraSample]
    scene_name:str
    
    @classmethod
    def from_nuscenes_sample(cls, nusc:NuScenes, sample_token:str):
        sample = nusc.get("sample", sample_token)
        timestamp = sample["timestamp"]
        annotations = []
        for ann_token in sample["anns"]:
            ann_data = nusc.get("sample_annotation", ann_token)
            category_name = ann_data["category_name"]
            instance_token = ann_data["instance_token"]
            visibility = ann_data["visibility_token"]
            ego_translation = tuple(ann_data["translation"])
            size = tuple(ann_data["size"])
            ego_rotation = tuple(ann_data["rotation"])
            next_ann = ann_data["next"]
            prev_ann = ann_data["prev"]
            ego_pose = to_transform(ego_translation, ego_rotation)
            annotation = Annotation(category_name, ann_token, instance_token, visibility, ego_pose, size, next_ann, prev_ann)
            annotations.append(annotation)
        
        cameras = {}
        for channel in CAMERA_CHANNELS:
            cam_image_token = sample["data"][channel.value]
            cam_sample_data = nusc.get("sample_data", cam_image_token)
            filename = cam_sample_data["filename"]
            prev_cam_image_token = cam_sample_data["prev"]
            next_cam_image_token = cam_sample_data["next"]
            is_key_frame = cam_sample_data["is_key_frame"]
            # car to world transformation
            ego_rotation = nusc.get("ego_pose",cam_sample_data["ego_pose_token"])["rotation"]
            ego_translation = nusc.get("ego_pose",cam_sample_data["ego_pose_token"])["translation"]
            ego_pose = to_transform(ego_translation, ego_rotation)
            # camera to car transformation
            sensor_translation = nusc.get("calibrated_sensor", cam_sample_data["calibrated_sensor_token"])["translation"]
            sensor_rotation = nusc.get("calibrated_sensor", cam_sample_data["calibrated_sensor_token"])["rotation"]
            sensor_pose = to_transform(sensor_translation, sensor_rotation)
            transformation = ego_pose * sensor_pose
            camera_intrinsics = np.array(nusc.get("calibrated_sensor", cam_sample_data["calibrated_sensor_token"])["camera_intrinsic"])
            camera_sample = CameraSample(
                image_token=cam_image_token,
                sample_token=sample_token,
                scene_token=sample["scene_token"],
                channel=channel.value,
                filename=filename,
                prev=prev_cam_image_token,
                next=next_cam_image_token,
                timestamp=cam_sample_data["timestamp"],
                transformation=transformation,
                is_key_frame=is_key_frame,
                camera_intrinsics=camera_intrinsics,
            )
            cameras[channel.value] = camera_sample
        
        scene_name = nusc.get("scene",sample["scene_token"])["name"]
        
        return cls(timestamp, annotations, cameras,scene_name)



def get_camera_sample_lists(
    nusc: NuScenes,
    scene: dict[str, Any],
    channels: tuple[str, ...] | None = None,
) -> dict[str, list[CameraSample]]:
    """Build an ordered sample_data chain for each requested camera channel."""
    if channels is None:
        channels = tuple(channel.value for channel in CAMERA_CHANNELS)

    first_sample = nusc.get("sample", scene["first_sample_token"])

    chains: dict[str, list[CameraSample]] = {}
    for channel in channels:
        if channel not in first_sample["data"]:
            continue

        sample_data = nusc.get("sample_data", first_sample["data"][channel])
        while sample_data["prev"]:
            sample_data = nusc.get("sample_data", sample_data["prev"])

        nodes: list[CameraSample] = []
        while True:
            file_name = str(sample_data.get("filename", ""))
            image_type = "sample" if "samples" in file_name else "sweep" #TODO make this more robust
            ego_pose_data = nusc.get("ego_pose", sample_data["ego_pose_token"])
            ego_pose: RigidTransform = to_transform(
                tuple(ego_pose_data["translation"]),
                tuple(ego_pose_data["rotation"]),
            )
            calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            sensor_pose: RigidTransform = to_transform(
                tuple(calibrated_sensor["translation"]),
                tuple(calibrated_sensor["rotation"]),
            )
            nodes.append(
                CameraSample(
                    image_token=str(sample_data["token"]),
                    sample_token=str(sample_data["sample_token"]),
                    channel=str(sample_data["channel"]),
                    filename=file_name,
                    prev=str(sample_data["prev"]),
                    next=str(sample_data["next"]),
                    transformation=ego_pose * sensor_pose,
                    is_key_frame=bool(sample_data["is_key_frame"]),
                    camera_intrinsics=np.array(calibrated_sensor["camera_intrinsic"]),
                    scene_token=str(first_sample["scene_token"]),
                    image_type=image_type,
                    timestamp=sample_data["timestamp"],
                )
            )
            if not sample_data["next"]:
                break
            sample_data = nusc.get("sample_data", sample_data["next"])

        # Second pass: link adjacent key frames via next_sample / prev_sample.
        # Sweeps keep the default None values.
        key_frame_indices = [i for i, n in enumerate(nodes) if n.is_key_frame]
        for pos, idx in enumerate(key_frame_indices):
            nodes[idx].prev_sample = (
                nodes[key_frame_indices[pos - 1]].image_token if pos > 0 else None
            )
            nodes[idx].next_sample = (
                nodes[key_frame_indices[pos + 1]].image_token
                if pos < len(key_frame_indices) - 1
                else None
            )

        chains[channel] = nodes

    return chains

def select_scenes(
    nusc: NuScenes,
    scene_names: list[str] | None,
    scene_tokens: list[str] | None,
    max_scenes: int | None,
) -> list[dict[str, Any]]:
    """Return scenes filtered by name/token and optionally truncated."""
    filtered = list(nusc.scene)

    if scene_names:
            wanted_names = set(scene_names)
            filtered = [scene for scene in filtered if scene["name"] in wanted_names]

    if scene_tokens:
            wanted_tokens = set(scene_tokens)
            filtered = [scene for scene in filtered if scene["token"] in wanted_tokens]

    filtered.sort(key=lambda row: row["name"])
    if max_scenes is not None:
            filtered = filtered[: max(0, max_scenes)]
    return filtered

def build_lookup_tables(nusc: NuScenes) -> tuple[dict[str, str], dict[str, str]]:
    """Build lookups for instance category names and attribute names."""
    category_name_by_token = {row["token"]: row["name"] for row in nusc.category}
    category_name_by_instance = {
            row["token"]: category_name_by_token.get(row["category_token"], "unknown")
            for row in nusc.instance
    }
    attribute_name_by_token = {row["token"]: row["name"] for row in nusc.attribute}
    return category_name_by_instance, attribute_name_by_token

