import numpy as np
import torch
import cv2
import math
import torch
import multiprocessing
import time

import argparse
import os
import json
import shutil
from tqdm import tqdm

import detectron2.structures as structures

from mass.utils.experimentation import TimeoutDueToUnityCrash
from mass.utils.experimentation import NumpyJSONEncoder
from mass.utils.experimentation import handle_read_only
from mass.utils.experimentation import run_experiment_with_restart

from mass.thor.segmentation_config import SegmentationConfig
from mass.thor.segmentation_config import ID_TO_OPENABLE, CLASS_TO_COLOR

def keep_gpu_busy():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting thread.")
        return

    # Adjust the following parameters based on your GPU's capacity
    tensor_size = (5000, 5000)  # Start with a medium-sized tensor
    num_operations = 65  # Start with a moderate number of operations

    while True:
        tensor = torch.randn(*tensor_size, device="cuda")
        for _ in range(num_operations):
            tensor = tensor * torch.randn(*tensor_size, device="cuda")
        
        # Short sleep to allow for some idle time
        time.sleep(0.2)
    pass

def get_object_name_from_id(category_id):
    # index = category_id - 1
    index = category_id
    object_name = list(CLASS_TO_COLOR.keys())[index]
    return object_name

def get_object_name_objectID(objectID):
    parts = objectID.split('|')
    object_name = parts[0]
    return object_name
    


def create_dataset(args):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """

    padding = 5
    dilate_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * padding + 1,
                            2 * padding + 1), (padding, padding))

    padding = 3
    erode_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * padding + 1,
                            2 * padding + 1), (padding, padding))

    # create arguments for the training and testing tasks
    with TimeoutDueToUnityCrash(2000):  # wait 300 seconds for unity
        task_params = SegmentationConfig.stagewise_task_sampler_args(
            stage=args.stage, process_ind=0, total_processes=1, devices=[0])

    task_params["ground_truth"] = True
    task_params["thor_controller_kwargs"] = \
        dict(renderInstanceSegmentation=True)

    # generate a sampler for training or testing evaluation
    with TimeoutDueToUnityCrash(2000):  # wait 300 seconds for unity
        task_sampler = SegmentationConfig.make_sampler_fn(
            **task_params, force_cache_reset=True,
            epochs=1, only_one_unshuffle_per_walkthrough=True)

    for task_id in range(args.start_task):
        with TimeoutDueToUnityCrash(2000):  # wait 60 seconds for unity to connect
            next(task_sampler.task_spec_iterator)

    # perform evaluation using every task in the task sampler
    for task_id in tqdm(range(args.start_task, args.start_task + (
            args.total_tasks * args.every_tasks), args.every_tasks)):

        def callback(obs: dict, image_index):
            """Callback function executed at every step during an episode
            that processes the current observation and serializes data
            for training a mask rcnn model at a later stage.

            Arguments:

            obs: dict
                the current observation from the environment, which contains
                ground truth semantic segmentation at the 'semantic' key.

            """

            image_id = (args.start_task *
                        args.images_per_task + image_index)


            # construct the disk path to the rgb and semantic images
            file_name = f"rgb/{image_id:07d}-rgb.png"  # we will generate
            sem_seg_file_name = f"sem/{image_id:07d}-sem.png"
            pan_seg_file_name = f"pan/{image_id:07d}-pan.png"
            depth_file_name = f"depth/{image_id:07d}-depth.png"

            # store semantic segmentation annotations in a dictionary format
            _postion = task.env.controller.last_event.metadata["agent"]["position"]
            _rotation = task.env.controller.last_event.metadata["agent"]["rotation"]
            _cameraHorizon = task.env.controller.last_event.metadata["agent"]["cameraHorizon"]
            _isStanding = task.env.controller.last_event.metadata["agent"]["isStanding"]
            training_example = dict(image_id=image_id, file_name=file_name,
                                    agent_coordinates=[_postion["x"], _postion["y"], _postion["z"]],
                                    agent_rotation=[_rotation["x"], _rotation["y"], _rotation["z"]],
                                    agent_cameraHorizon=_cameraHorizon,
                                    agent_isStanding=_isStanding,
                                    sem_seg_file_name=sem_seg_file_name,
                                    pan_seg_file_name=pan_seg_file_name,
                                    depth_file_name=depth_file_name,
                                    height=obs["semantic"].shape[0],
                                    width=obs["semantic"].shape[1],
                                    segments_info=[], annotations=[])
            
            # print("agent_cameraHorizon: ", training_example["agent_cameraHorizon"])
            # print("agent_isStanding: ", training_example["agent_isStanding"])
            # print("agent_coordinates: ", training_example["agent_coordinates"])
            # print("agent_rotation: ", training_example["agent_rotation"])
            
            # create a buffer to store instance ids for every object
            total_instances = 0  # differs from the ground truth instances
            instances = np.zeros(obs["semantic"].shape[:2], dtype=np.int32)
            training_example["segments_info"].append(
                dict(category_id=0, id=0, isthing=False))

            # generate a set of instance ids from the environment
            instance_segmentation = (  # to handle partial occlusion
                task.env.last_event.instance_segmentation_frame)
            
            # print("instance_segmentation: ", instance_segmentation)

            # depth = (  # to handle partial occlusion
            #     task.env.last_event.depth_frame)
            # print("depth: ", depth)
            instance_segmentation = (instance_segmentation[..., 0:1] +
                                     instance_segmentation[..., 1:2] * 256 +
                                     instance_segmentation[..., 2:3] * 256 * 256)

            # iterate through every detected class except the background
            for category_id in (set(  # and generate segmentation annotations
                    np.unique(obs["semantic"]).tolist()) - {0}):  # per class

                # generate a mask that shows where category_id is
                semantic_mask = (category_id ==  # located in the image
                                 obs["semantic"]).astype(np.uint8)

                # iterate through each object instance in the frame
                for instance_id in (np.unique(  # and generate annotations
                        instance_segmentation[
                            np.nonzero(semantic_mask)]).tolist()):

                    # generate a mask that shows where category_id is
                    instance_mask = (instance_id ==  # located in the image
                                     instance_segmentation).astype(np.uint8)
                    instance_mask = cv2.erode(cv2.dilate(
                        instance_mask[..., 0], dilate_element), erode_element)

                    # for this object instance generate a set of polygons
                    pts = cv2.findContours(instance_mask,
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)[-2]

                    pts = [x for x in pts if x.shape[0] > 2]
                    if len(pts) == 0:  # skip instances that are too small
                        continue

                    # record the current object as a detected instance for
                    total_instances += 1  # panoptic segmentation in detectron
                    cv2.fillPoly(instances, pts=pts, color=total_instances)
                    training_example["segments_info"].append(
                        dict(category_id=category_id,
                             id=total_instances, isthing=True))
                    
                    # print("category_id: ", category_id)
                    object_name_from_frame = get_object_name_from_id(category_id)

                
                    # Find the corresponding object in the last event metadata to get the openness
                    openness = None
                    object_oriented_bounding_box = None
                    for obj in task.env.controller.last_event.metadata["objects"]:
                        object_oriented_bounding_box = obj["objectOrientedBoundingBox"]

                        # get openness value
                        object_name_from_objectID = get_object_name_objectID(obj["objectId"])
                        # print("object_name_from_frame: ", object_name_from_frame)
                        # print("object_name_from_objectID: ", object_name_from_objectID)
                        if object_name_from_objectID == object_name_from_frame:
                            openness = obj["openness"]
                            # print("openness: ", openness)
                            break  # Once the object is found, no need to continue the loop

                    # record the location of the detected object instance
                    # in the frame with a bounding box, mask, and category_id
                    training_example["annotations"].append(
                        dict(bbox=cv2.boundingRect(np.concatenate(pts, axis=0)),
                             bbox_mode=structures.BoxMode.XYWH_ABS,
                             category_id=category_id, 
                             openness=openness,  # Add the openness attribute
                             object_oriented_bounding_box = object_oriented_bounding_box, # Add the object oriented bounding boxs attribute
                             segmentation=[
                                x.flatten().tolist() for x in pts]))

            # construct an rgb image for panoptic segmentation following
            instance_rgb = np.zeros([instances.shape[0],  # the coco api
                                     instances.shape[1], 3], dtype=np.uint8)
            for idx in range(3):
                instance_rgb[..., idx] = (instances // (256 ** idx)) % 256

            if len(training_example["annotations"]) == 0:
                return 0  # images without objects are not used by detectron2

            # write the observed image and its generated segmentation labels
            # to the disk for training mask r-cnn with detectron
            cv2.imwrite(os.path.join(args.logdir, file_name),
                        255 * obs["rgb"][..., ::-1])
            cv2.imwrite(os.path.join(args.logdir, sem_seg_file_name),
                        obs["semantic"][..., 0])
            cv2.imwrite(os.path.join(args.logdir, pan_seg_file_name),
                        instance_rgb[..., ::-1])
            cv2.imwrite(os.path.join(args.logdir, depth_file_name),
                        255 * obs["depth"][..., ::-1])

            # open the target file for this image and write
            with open(os.path.join(  # the annotations to a json file
                    args.logdir, f"annotations/{image_id:07d}.json"), "w") as f:
                json.dump(training_example, f,
                          indent=4, cls=NumpyJSONEncoder)

            return 1

        with TimeoutDueToUnityCrash():
            task = task_sampler.next_task()

        print("TaskID: {}, Task Scene: {}".format(task_id, task.env.controller.last_event.metadata["sceneName"]))


        valid_positions = task.env.controller.step(
            action="GetReachablePositions").metadata["actionReturn"]

        valid_positions = [dict(position=position,
                                rotation=dict(x=0, y=rotation, z=0),
                                horizon=horizon, standing=standing)
                           for position in valid_positions
                           for rotation in (0, 90, 180, 270)
                           for horizon in (-30, 0, 30, 60)
                           for standing in (True, False)]

        valid_positions = [valid_positions[idx] for idx in
                           np.random.permutation(len(valid_positions))]

        num_images = 0

        while num_images < args.images_per_task // 2:
            task.env.controller.step(action="TeleportFull", **valid_positions.pop(0))
            num_images += callback(task.get_observations(), num_images)

        with TimeoutDueToUnityCrash():
            task = task_sampler.next_task()

        valid_positions = task.env.controller.step(
            action="GetReachablePositions").metadata["actionReturn"]

        valid_positions = [dict(position=position,
                                rotation=dict(x=0, y=rotation, z=0),
                                horizon=horizon, standing=standing)
                           for position in valid_positions
                           for rotation in (0, 90, 180, 270)
                           for horizon in (-30, 0, 30, 60)
                           for standing in (True, False)]

        valid_positions = [valid_positions[idx] for idx in
                           np.random.permutation(len(valid_positions))]

        while num_images < args.images_per_task:
            task.env.controller.step(action="TeleportFull", **valid_positions.pop(0))
            num_images += callback(task.get_observations(), num_images)

        for _ in range(args.every_tasks - 1):
            with TimeoutDueToUnityCrash():
                next(task_sampler.task_spec_iterator)

        args.total_tasks -= 1

        args.start_task += args.every_tasks


def run_experiment(args):
    """Semantic Mapping agent for solving AI2-THOR Rearrangement in PyTorch,
    by building two semantic maps, one for the goal state of the world,
    and a second for the current world state, and tracking disagreement.

    """

    name = (f"{args.start_task}-"
            f"{args.start_task + args.total_tasks * args.every_tasks}")

    os.makedirs(os.path.join(args.logdir, f"annotations"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"images"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"pan"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"sem"), exist_ok=True)
    os.makedirs(os.path.join(args.logdir, f"depth"), exist_ok=True)

    os.makedirs(os.path.join(args.logdir, f"tmp-{name}"), exist_ok=True)

    proc = multiprocessing.Process(target=keep_gpu_busy)
    proc.start()

    os.environ["HOME"] = os.path.join(args.logdir, f"tmp-{name}")

    run_experiment_with_restart(create_dataset, args)

    shutil.rmtree(os.environ["HOME"], onerror=handle_read_only)
    
    proc.terminate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Data Collection Agent")

    parser.add_argument("--logdir", type=str,
                        default="/scratch/xt2191/Rearrangement_LLM/test-data-openness")

    parser.add_argument("--stage",
                        type=str, default="train")
    parser.add_argument("--start-task",
                        type=int, default=0)
    parser.add_argument("--every-tasks",
                        type=int, default=1)
    parser.add_argument("--total-tasks",
                        type=int, default=4000)

    parser.add_argument("--images-per-task",
                        type=int, default=200)

    run_experiment(parser.parse_args())
