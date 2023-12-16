import cv2
import numpy as np
import json
import os

import math
from Rearrangement_LLM.scripts.test_maskrcnn_demo2 import generate_2dboxes

def to_rad(deg):
    return deg * np.pi / 180


def load_images(rgb_image_path, depth_image_path):
    rgb_image = cv2.imread(rgb_image_path)

    # Read depth information from .png file
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    # Transform depth values back to the real depth values
    depth_normalized = depth_image / 255.0

    return rgb_image, depth_normalized

def object_detection(rgb_image):
    boxes = generate_2dboxes(rgb_image)
    print("2D: \n", boxes)
    return boxes


def create_rotation_matrix(horizon, pitch, yaw, roll):
    # Camera rotation matrix
    R_horizon = np.array([[1, 0, 0],
                          [0, np.cos(to_rad(horizon)), -np.sin(to_rad(horizon))],
                          [0, np.sin(to_rad(horizon)), np.cos(to_rad(horizon))]])

    # Agent rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(to_rad(pitch)), -np.sin(to_rad(pitch))],
                    [0, np.sin(to_rad(pitch)), np.cos(to_rad(pitch))]])
    R_y = np.array([[np.cos(to_rad(yaw)), 0, np.sin(to_rad(yaw))],
                    [0, 1, 0],
                    [-np.sin(to_rad(yaw)), 0, np.cos(to_rad(yaw))]])
    R_z = np.array([[np.cos(to_rad(roll)), -np.sin(to_rad(roll)), 0],
                    [np.sin(to_rad(roll)), np.cos(to_rad(roll)), 0],
                    [0, 0, 1]])

    # Combine rotation matrices
    R_body = np.dot(R_z, np.dot(R_y, R_x))  # Agent rotation matrix
    R = np.dot(R_body, R_horizon)  # Final rotation matrix - Camera

    return R



def camera_to_world(depth_image, camera_coord, camera_pose, fov=90):
    """
    Convert camera coordinates to world coordinates.

    :param depth_image: A depth image where each pixel value represents distance to the camera.
    :param camera_coord: Camera coordinates in the world.
    :param camera_pose: Camera orientation (rotation matrix or quaternion).
    :param fov: Field of View of the camera in degrees (assuming a default value).
    :return: A set of 3D points in the world coordinate system.
    """
    height, width = depth_image.shape

    # Approximate focal length
    # f = width / (2 * np.tan(fov * np.pi / 360))
    f = 0.5 * width / math.tan(to_rad(fov/2))

    # Get pixel coordinates
    cx, cy = width // 2, height // 2
    x = np.linspace(0, width - 1, width) - cx
    y = np.linspace(0, height - 1, height) - cy
    x, y = np.meshgrid(x, -y)

    # Convert depth informaation to 3D coordinates 
    z = depth_image
    x = np.multiply(x, z) / f
    y = np.multiply(y, z) / f

    # Transform camera coordinates to world coordinates
    points_3d = np.dstack((x, y, z))
    points_3d = points_3d.reshape(-1, 3)
    points_world = np.dot(points_3d, camera_pose.T) + camera_coord

    # print("points_world: \n", points_world)

    return points_world

def extract_depth_from_box(depth_image, box):
    xmin, ymin, xmax, ymax = box.astype(int)
    return depth_image[ymin:ymax, xmin:xmax]


def estimate_3d_bounding_boxes(bounding_boxes_2d, depth_image, camera_coord, camera_pose, fov=90):
    bounding_boxes_3d = []

    for box in bounding_boxes_2d:
        depth_box = extract_depth_from_box(depth_image, box)
        if depth_box.size == 0:
            continue

        world_points = camera_to_world(depth_box, camera_coord, camera_pose, fov)
        min_coord = world_points.min(axis=0)
        max_coord = world_points.max(axis=0)
        # Calculate 3D bounding box
        bbox_3d = np.array([
            [min_coord[0], min_coord[1], min_coord[2]],
            [max_coord[0], min_coord[1], min_coord[2]],
            [min_coord[0], max_coord[1], min_coord[2]],
            [max_coord[0], max_coord[1], min_coord[2]],
            [min_coord[0], min_coord[1], max_coord[2]],
            [max_coord[0], min_coord[1], max_coord[2]],
            [min_coord[0], max_coord[1], max_coord[2]],
            [max_coord[0], max_coord[1], max_coord[2]],
        ])
        bounding_boxes_3d.append(bbox_3d)

    return bounding_boxes_3d


# read GT from json file
# TODO: some GT is null
def read_bounding_box_from_file(file_path):
    bounding_boxes = []

    if os.path.isfile(file_path) and file_path.endswith(".json"):
        with open(file_path, 'r') as file:
            data = json.load(file)

            if 'annotations' in data:
                for annotation in data['annotations']:
                    # 确保 object_oriented_bounding_box 存在且不为 None
                    if annotation.get('object_oriented_bounding_box'):
                        oobb = annotation['object_oriented_bounding_box']
                        # 检查 cornerPoints 是否存在
                        if 'cornerPoints' in oobb:
                            bounding_box = oobb['cornerPoints']
                            bounding_boxes.append(bounding_box)

    return bounding_boxes






# compute 3D IOU
def compute_3d_iou(box1, box2):
    # 将每个3D边界框转换为轴对齐的边界框
    def get_aabb(box):
        min_coord = np.min(box, axis=0)
        max_coord = np.max(box, axis=0)
        return min_coord, max_coord

    box1_min, box1_max = get_aabb(box1)
    box2_min, box2_max = get_aabb(box2)

    # 计算每个维度上的最大最小值
    xmin = max(box1_min[0], box2_min[0])
    ymin = max(box1_min[1], box2_min[1])
    zmin = max(box1_min[2], box2_min[2])
    xmax = min(box1_max[0], box2_max[0])
    ymax = min(box1_max[1], box2_max[1])
    zmax = min(box1_max[2], box2_max[2])

    # 计算交集体积
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin) * max(0, zmax - zmin)

    # 计算每个框的体积
    volume1 = np.prod(box1_max - box1_min)
    volume2 = np.prod(box2_max - box2_min)

    # 计算并集体积
    union = volume1 + volume2 - intersection

    # 计算IOU
    iou = intersection / union if union != 0 else 0

    return iou


# 示例使用
rgb_image_path = '/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/test_data_dir/rgb/0000019-rgb.png'
depth_image_path = '/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/test_data_dir/depth/0000019-depth.png'
json_file_path = '/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/test_data_dir/annotations/0000019.json'  

rgb_image, depth_image = load_images(rgb_image_path, depth_image_path)

# 假设相机坐标和姿态
agent_coord = np.array([-0.75, 0.9014922380447388, -1.25])  # 示例坐标
horizon = -30.0
agent_rotation = np.array([-0.0, 90.0, 0.0])
pitch, yaw, roll = agent_rotation[0], agent_rotation[1], agent_rotation[2]         # 示例旋转角度
camera_pose = create_rotation_matrix(horizon, pitch, yaw, roll)

# Get estimate 3D bb
bb_3d = estimate_3d_bounding_boxes(object_detection(rgb_image), depth_image, agent_coord, camera_pose)
print("3D: \n", bb_3d)

# Get GT 3D bb
GT_bb_3d = read_bounding_box_from_file(json_file_path)
print("GT 3D: \n", GT_bb_3d)

# Compute 3D IOU
# TODO: match the obejects
iou = compute_3d_iou(bb_3d[0], GT_bb_3d[0])
print("IOU: \n", iou)

