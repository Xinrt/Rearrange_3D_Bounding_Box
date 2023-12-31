import torch
import torch.nn.functional as functional
import cv2
import numpy as np
from mass.nn.base_projection_layer import BaseProjectionLayer
from typing import Dict, Any


class SemanticProjectionLayer(BaseProjectionLayer):
    """Create a feature projection layer in PyTorch that maintains a voxel
    grid description of the world world, where each voxel grid cell has
    a feature vector associated with it, typically semantic labels.

    Arguments:

    camera_height: int
        the map_height of the image generated by a pinhole camera onboard the
        agent, corresponding to a map_depth and semantic observation.
    camera_width: int
        the map_width of the image generated by a pinhole camera onboard the
        agent, corresponding to a map_depth and semantic observation.
    vertical_fov: float
        the vertical field of view of the onboard camera, measure in
        radians from the bottom of the viewport to the top.

    map_height: int
        the number of grid cells along the 'map_height' axis of the semantic
        map, as rendered using the top down rendering function.
    map_width: int
        the number of grid cells along the 'map_width' axis of the semantic
        map, as rendered using the top down rendering function.
    map_depth: int
        the number of grid cells that are collapsed along the 'up'
        direction by the top down rendering function.
    feature_size: int
        the number of units in each feature vector associated with every
        grid cell, such as the number of image segmentation categories.

    origin_y: float
        the center of the semantic map along the 'map_height' axis of the
        semantic map as viewed from a top-down render of the map.
    origin_x: float
        the center of the semantic map along the 'map_width' axis of the
        semantic map as viewed from a top-down render of the map.
    origin_z: float
        the center of the semantic map along the 'map_depth' axis of the
        semantic map as viewed from a top-down render of the map.
    grid_resolution: float
        the length of a single side of each voxel in the semantic map in
        units of the world coordinate system, which is typically meters.

    interpolation_weight: float
        float representing the interpolation weight used when adding
        new features in the feature map weighted by interpolation_weight.
    initial_feature_map: torch.Tensor
        tensor representing the initial feature map tensor,
        which will be set to zero if the value not specified by default.
    class_to_colors: torch.Tensor
        a tensor representing a mapping from object categories in the
        semantic map to rgb values for visualization.

    """

    def __init__(self, camera_height: int = 224, camera_width: int = 224,
                 vertical_fov: float = 90.0, map_height: int = 256,
                 map_width: int = 256, map_depth: int = 64,
                 feature_size: int = 1, dtype: torch.dtype = torch.float32,
                 origin_y: float = 0.0, origin_x: float = 0.0,
                 origin_z: float = 0.0, grid_resolution: float = 0.05,
                 interpolation_weight: float = 0.5,
                 initial_feature_map: torch.Tensor = None,
                 class_to_colors: torch.Tensor = None):
        """Create a feature projection layer in PyTorch that maintains a voxel
        grid description of the world world, where each voxel grid cell has
        a feature vector associated with it, typically semantic labels.

        Arguments:

        camera_height: int
            the map_height of the image generated by a pinhole camera onboard the
            agent, corresponding to a map_depth and semantic observation.
        camera_width: int
            the map_width of the image generated by a pinhole camera onboard the
            agent, corresponding to a map_depth and semantic observation.
        vertical_fov: float
            the vertical field of view of the onboard camera, measure in
            radians from the bottom of the viewport to the top.

        map_height: int
            the number of grid cells along the 'map_height' axis of the semantic
            map, as rendered using the top down rendering function.
        map_width: int
            the number of grid cells along the 'map_width' axis of the semantic
            map, as rendered using the top down rendering function.
        map_depth: int
            the number of grid cells that are collapsed along the 'up'
            direction by the top down rendering function.
        feature_size: int
            the number of units in each feature vector associated with every
            grid cell, such as the number of image segmentation categories.

        origin_y: float
            the center of the semantic map along the 'map_height' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_x: float
            the center of the semantic map along the 'map_width' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_z: float
            the center of the semantic map along the 'map_depth' axis of the
            semantic map as viewed from a top-down render of the map.
        grid_resolution: float
            the length of a single side of each voxel in the semantic map in
            units of the world coordinate system, which is typically meters.

        interpolation_weight: float
            float representing the interpolation weight used when adding
            new features in the feature map weighted by interpolation_weight.
        initial_feature_map: torch.Tensor
            tensor representing the initial feature map tensor,
            which will be set to zero if the value not specified by default.
        class_to_colors: torch.Tensor
            a tensor representing a mapping from object categories in the
            semantic map to rgb values for visualization.

        """

        super(SemanticProjectionLayer, self).__init__(
            camera_height=camera_height, camera_width=camera_width,
            vertical_fov=vertical_fov, map_height=map_height,
            map_width=map_width, map_depth=map_depth,
            feature_size=feature_size, dtype=dtype,
            origin_y=origin_y, origin_x=origin_x,
            origin_z=origin_z, grid_resolution=grid_resolution,
            interpolation_weight=interpolation_weight,
            initial_feature_map=initial_feature_map)

        self.boxes = None
        self.register_buffer('class_to_colors', torch.as_tensor(
            class_to_colors, dtype=torch.float32, device=self.data.device))

    def reset(self, origin_y: float = 0.0,
              origin_x: float = 0.0, origin_z: float = 0.0):
        """Utility function for clearing the contents of the feature map,
        which is typically called at the beginning of an episode with
        a new map origin, in order to reduce gpu memory usage.

        Arguments:

        origin_y: float
            the center of the semantic map along the 'map_height' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_x: float
            the center of the semantic map along the 'map_width' axis of the
            semantic map as viewed from a top-down render of the map.
        origin_z: float
            the center of the semantic map along the 'map_depth' axis of the
            semantic map as viewed from a top-down render of the map.

        """

        self.boxes = None
        super(SemanticProjectionLayer, self).reset(
            origin_y=origin_y, origin_x=origin_x, origin_z=origin_z)

    def update(self, observation: Dict[str, torch.Tensor]):
        """Update the semantic map given a map_depth image and a feature image
        by projecting the features onto voxels in the semantic map using
        a set of rays emanating from a virtual pinhole camera.

        Arguments:

        observation["position"]: torch.Tensor
            the position of the agent in the world coordinate system, where
            the position will be binned to voxels in a semantic map.
        observation["yaw"]: torch.Tensor
            a tensor representing the yaw in radians of the coordinate,
            starting from the x-axis and turning counter-clockwise.
        observation["elevation"]: torch.Tensor
            a tensor representing the elevation in radians about the x-axis,
            with positive corresponding to upwards tilt.

        observation["map_depth"]: torch.FloatTensor
            the length of the corresponding ray in world coordinates before
            hitting a surface, with shape: [height, width, 1].
        observation["features"]: Any
            a feature vector for every pixel in the imaging plane, to be
            scattered on the map, with shape: [height, width, num_classes].

        """

        # ensure all tensors have the appropriate device and dtype
        position = torch.as_tensor(observation[
            "position"], dtype=torch.float32, device=self.data.device)
        yaw = torch.as_tensor(observation[
            "yaw"], dtype=torch.float32, device=self.data.device)
        elevation = torch.as_tensor(observation[
            "elevation"], dtype=torch.float32, device=self.data.device)
        depth = torch.as_tensor(observation[
            "depth"], dtype=torch.float32, device=self.data.device)

        # grab a semantic segmentation observation, and encode it as a
        # one hot categorical distribution per image pixel
        semantic = torch.LongTensor(
            observation["semantic"]).to(device=self.data.device)[..., 0]

        # grab a semantic segmentation observation, and encode it as a
        # one hot categorical distribution per image pixel
        semantic = functional.one_hot(semantic,
                                      num_classes=self.feature_size)

        # update the semantic feature map using the latest observations
        super(SemanticProjectionLayer, self).update(  # from the environment
            dict(position=position, yaw=yaw, elevation=elevation,
                 depth=depth, features=semantic.to(torch.float32)))

        return self  # return self for chaining additional functions

    def visualize(self, obs: Dict[str, Any],
                  depth_slice: slice = slice(0, 32)):
        """Helper function that returns a list of images that are used for
        visualizing the contents of the feature map contained in subclasses,
        such as visualizing object categories, or which voxels are obstacles.

        Arguments:

        obs: Dict[str, dict]
            the current observation, as a dict or Tensor, which can be
            used to visualize the current location of the agent in the scene.
        depth_slice: Union[slice, Dict[str, slice]]
            an slice that specifies which map_depth components to use
            when rendering a top down visualization of the feature map.

        Returns:

        image: np.ndarray
            a list of numpy arrays that visualize the contents of this layer,
             such as an image showing semantic categories.

        """

        # generate a top down projection of the class probabilities
        top_down_map = self.top_down(depth_slice=depth_slice)

        # lookup a correspond color for each semantic category
        image = torch.nn.functional.embedding(
            top_down_map.argmax(dim=-1), self.class_to_colors)
        image = torch.where((top_down_map != 0).any(dim=-1, keepdim=True),
                            image, torch.ones_like(image)).cpu().numpy()

        # if we have previously called find this episode
        if self.boxes is not None:  # visualize where objects were
            for x, y, w, h in self.boxes:  # there can be multiple objects
                cv2.rectangle(image, (x, y), (x + w, y + h), (1, 0, 0), 1)

        return image  # top down visualization of the semantic map

    def find(self, semantic_category: int, confidence_threshold: float = 0.2,
             contour_padding: int = 3, contour_threshold: float = 0.0,
             feature_map = None):
        """Localize a semantic category in the map by computing the expected
        position of that object class in the semantic map, and return the
        confidence of the prediction via the expected class probability.

        Arguments:

        semantic_category: int
            an integer that represent which semantic category to localize
            in the semantic map by calculating the expected position.
        confidence_threshold: float
            a threshold for confidence that determines the minimum
            confidence a detection must have to be considered a positive.
        contour_padding: int
            an integer representing the radius of the kernel used for
            smoothing the semantic map when counting the number of objects.
        contour_threshold: float
            a threshold used to determine whether a pixel in the class map
            is considered to be a part of a contour by opencv.

        Returns:

        confidence: List[torch.Tensor]
            tensor list containing the counts of number occupied voxels in
            the semantic map for each object segmentation class.
        coordinate: List[torch.Tensor]
            tensor list representing the expected location in map coordinates
            for every detected object segmented in the semantic map.
        sizes: List[torch.Tensor]
            tensor list representing the expected size in voxels for every
            detected object segmented in the semantic map.

        """

        # ensure that all new tensors are on the same device
        kwargs = dict(dtype=torch.float32, device=self.data.device)

        # construct a set of indices into the grid of voxels in the semantic
        # map, used to computing the locations of object in map coordinates
        y, x, z = torch.meshgrid(torch.arange(self.map_height, **kwargs),
                                 torch.arange(self.map_width, **kwargs),
                                 torch.arange(self.map_depth,
                                              **kwargs), indexing='ij')

        # slice the semantic map given a semantic_category
        mask = self.data[..., semantic_category:semantic_category + 1]
        mask_coordinates = self.map_to_world(torch.stack([x, y, z], dim=-1))

        # smooth the semantic mask by convolving the mask with
        # an averaging filter with a kernel size of (padding * 2 + 1) pixels
        smooth = mask.permute(3, 0, 1, 2).unsqueeze(0)
        smooth = functional.avg_pool3d(smooth, contour_padding * 2 + 1,
                                       stride=1, padding=contour_padding)
        smooth = smooth.squeeze(0).permute(1, 2, 3, 0)

        # construct an image over the planar axes of the semantic map
        # representing which pixels contain an object of the selected class
        threshold_image = (smooth > contour_threshold)\
            .any(dim=2).detach().cpu().numpy().astype(np.uint8)

        self.boxes, coordinates, confidences, sizes, features = [], [], [], [], []

        # detect separate clusters of object pixels, which can be used
        # to count the number of objects in the scene of a particular class
        for contour in cv2.findContours(threshold_image[:, :, 0],
                                        cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)[0]:

            # detect a bounding box from the top down for this object
            x, y, w, h = cv2.boundingRect(contour)

            # select a region from the semantic map for this object
            mask_roi = mask[y:y + h, x:x + w]
            coords_roi = mask_coordinates[y:y + h, x:x + w]
            if feature_map is not None:
                features_roi = feature_map.data[
                    y:y + h, x:x + w].to(mask_roi.device)

            # convex combination weights for expected position
            weights = mask_roi / (mask_roi.sum() + 1e-9)
            detection_confidence = (mask_roi * weights).sum()

            if detection_confidence > confidence_threshold:

                # store the bounding box for visualizing detections
                self.boxes.append((x, y, w, h))  # and debugging purposes

                # compute the expected position and confidence
                # of this detection using the map class probabilities
                confidences.append(detection_confidence)
                coordinates.append((coords_roi *
                                    weights).sum(dim=(0, 1, 2)))

                # compute the number of voxels that are occupied by
                sizes.append(mask_roi.sum())  # the object in expectation

                # compute the number of voxels that are occupied by
                if feature_map is not None:
                    features.append((features_roi * weights).sum(dim=(0, 1, 2)))

        # return the expected position for the chosen class
        # the confidence of the detection (expected probability)
        return confidences, coordinates, sizes, \
            features if feature_map is not None else None
