import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage


import math
from YOLO_detection import generate_2dboxes
from mass.thor.segmentation_config import CLASS_TO_COLOR
from Dataset.Dataset import MyDataset, custom_collate_fn


DATASET_PATH = '/vast/xt2191/dataset'


def median_filter_depth_image(depth_image, kernel_size=5):
    """
    Apply a median filter to the depth image to reduce noise.
    """
    return scipy.ndimage.median_filter(depth_image, size=kernel_size)

def extract_median_depth_from_box(depth_image, box):
    """
    Extract the median depth value from the specified bounding box.
    """
    xmin, ymin, xmax, ymax = box.astype(int)
    depth_box = depth_image[ymin:ymax, xmin:xmax]
    if depth_box.size == 0:
        return None
    return np.median(depth_box[depth_box > 0])  # Exclude zero depth values



# Function to find the corresponding ids for given key names
def find_indexes_for_keys(keys, dictionary):
    indexes = {key: list(dictionary.keys()).index(key) for key in keys if key in dictionary}
    return indexes

def to_rad(deg):
    return deg * np.pi / 180

def object_detection(rgb_image):
    boxes, names = generate_2dboxes(rgb_image)
    print("2D: \n", boxes)
    return boxes, names


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


def estimate_3d_bounding_boxes(bounding_boxes_2d, index_boxes, depth_image, camera_coord, camera_pose, fov=90):
    bounding_boxes_3d = {}

    for box, idx in zip(bounding_boxes_2d, index_boxes):
        depth_box = extract_depth_from_box(depth_image, box)
        if depth_box.size == 0:
            continue

        world_points = camera_to_world(depth_box, camera_coord, camera_pose, fov)
        min_coord = world_points.min(axis=0)
        max_coord = world_points.max(axis=0)

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
        bounding_boxes_3d[idx] = bbox_3d


    return bounding_boxes_3d


def compute_3d_iou(box1, box2):
    # Convert each 3D bounding box to a shaft-aligned bounding box
    def get_aabb(box):
        min_coord = np.min(box, axis=0)
        max_coord = np.max(box, axis=0)
        return min_coord, max_coord

    box1_min, box1_max = get_aabb(box1)
    box2_min, box2_max = get_aabb(box2)

    # Calculate the maximum and minimum values on each dimension
    xmin = max(box1_min[0], box2_min[0])
    ymin = max(box1_min[1], box2_min[1])
    zmin = max(box1_min[2], box2_min[2])
    xmax = min(box1_max[0], box2_max[0])
    ymax = min(box1_max[1], box2_max[1])
    zmax = min(box1_max[2], box2_max[2])

    # Computed intersection volume
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin) * max(0, zmax - zmin)

    # Compute the volume of each 3D bounding box
    volume1 = np.prod(box1_max - box1_min)
    volume2 = np.prod(box2_max - box2_min)

    # Compute the union volume
    union = volume1 + volume2 - intersection

    # Compute the intersection over union
    iou = intersection / union if union != 0 else 0

    return iou


if __name__ == "__main__":

    dataset = MyDataset(DATASET_PATH)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

    print("Length: ", len(dataset))

    for batch in dataloader:
        annotations = batch['annotation']
        rgb_image = batch['rgb']
        depth_image = batch['depth'] / 255.0 

        for annotation in annotations:
            print("annotation: ", annotation)
            agent_coordinates = annotation['agent_coordinates']
            agent_cameraHorizon = annotation['agent_cameraHorizon']
            agent_rotation = annotation['agent_rotation']

            agent_coord = np.array(agent_coordinates)  
            horizon = agent_cameraHorizon
            agent_rotation = np.array(agent_rotation)

            pitch, yaw, roll = agent_rotation[0], agent_rotation[1], agent_rotation[2]         
            camera_pose = create_rotation_matrix(horizon, pitch, yaw, roll)


            # Get GT 3D bb
            bounding_boxes = {}
            annotation_in_file = annotation['annotations']
            # print("annotation_in_file: ", annotation_in_file)
            for annot in annotation_in_file:
                oobb = annot.get('object_oriented_bounding_box')
                if oobb and 'cornerPoints' in oobb:
                    obj_id = annot['category_id']
                    bounding_box = oobb['cornerPoints']
                    bounding_boxes[obj_id] = bounding_box

            print("GT 3D: \n", bounding_boxes)


        # Get estimate 3D bb
        # Ensure rgb_image is on CPU and convert to numpy
        rgb_numpy = rgb_image.squeeze(0).permute(1, 2, 0).cpu().numpy()

        boxes_2d, obj_names = object_detection(rgb_numpy)

        # 在这里添加边界框绘制代码
        fig, ax = plt.subplots(1)
        ax.imshow(rgb_numpy)
        for box in boxes_2d:
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        # 保存图像到文件
        output_path = "/scratch/xt2191/Rearrange_3D_Bounding_Box/boxes/utilsdetected_objects.png"  # 您可以修改这个路径为您希望的保存位置
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)


        ids_for_keys = find_indexes_for_keys(obj_names, CLASS_TO_COLOR)
        indexes_only = list(ids_for_keys.values())
        # print(indexes_only)



        depth_numpy = depth_image.squeeze().cpu().numpy()
        bb_3d = estimate_3d_bounding_boxes(boxes_2d, indexes_only, depth_numpy, agent_coord, camera_pose)
        print("3D: \n", bb_3d)



        ious = []
        for idx in indexes_only:
            if idx in bounding_boxes and idx in bb_3d:
                gt_box = bounding_boxes[idx]
                pred_box = bb_3d[idx]
                iou = compute_3d_iou(gt_box, pred_box)
                ious.append(iou)
                print(f"IOU for object ID {idx}: {iou}")
            else:
                print(f"Bounding box for object ID {idx} not found in both GT and predicted")

        # Average IOU
        if ious:
            avg_iou = sum(ious) / len(ious)
            print("Average IOU:", avg_iou)
        else:
            print("No IOU calculated")

        import pdb; pdb.set_trace()