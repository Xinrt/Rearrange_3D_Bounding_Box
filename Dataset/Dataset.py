import torch
import cv2
import os
import json


def read_image(file_path: str, color_mode: int) -> torch.Tensor:
    """
    Reads an image from the file path and converts it to a PyTorch tensor.
    """
    image = cv2.imread(file_path, color_mode)
    if image is None:
        raise FileNotFoundError(f"Image not found: {file_path}")

    if color_mode == cv2.IMREAD_COLOR:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return torch.from_numpy(image.transpose(2, 0, 1))


class Dataset(torch.utils.data.Dataset):
    ANNOTATION_FOLDER = 'annotation'
    RGB_FOLDER = 'rgb'
    DEPTH_FOLDER = 'depth'
    SEGMENTATION_FOLDER = 'seg'
    PAN_FOLDER = 'pan'

    def __init__(self, base_path: str):
        """

        :param base_path: The path to the dataset
        """
        self.base_path = base_path

        self.annotation_path = os.path.join(self.base_path, self.ANNOTATION_FOLDER)
        self.rgb_path = os.path.join(self.base_path, self.RGB_FOLDER)
        self.depth_path = os.path.join(self.base_path, self.DEPTH_FOLDER)
        self.segmentation_path = os.path.join(self.base_path, self.SEGMENTATION_FOLDER)
        self.pan_path = os.path.join(self.base_path, self.PAN_FOLDER)

        self.data_names = None
        self._load_and_validate_dataset()

    def _load_and_validate_dataset(self):
        """
        Loads the dataset file name from the base path,
        and assert they both have the same length in different folders
        """

        # use annotation folder to get the file names
        annotation_files_without_extension = list(map(
            lambda x: x.split('.')[0],
            os.listdir(self.annotation_path)
        ))

        length = len(annotation_files_without_extension)

        # use this length to check if the other folders have the same length
        assert len(os.listdir(self.rgb_path)) == length
        assert len(os.listdir(self.depth_path)) == length
        assert len(os.listdir(self.segmentation_path)) == length
        assert len(os.listdir(self.pan_path)) == length

        self.data_names = annotation_files_without_extension

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index: int):
        file_name = self.data_names[index]
        paths = {
            'annotation': os.path.join(self.base_path, 'annotation', file_name + '.json'),
            'rgb': os.path.join(self.base_path, 'rgb', file_name + '.jpg'),
            'depth': os.path.join(self.base_path, 'depth', file_name + '.jpg'),
            'segmentation': os.path.join(self.base_path, 'seg', file_name + '.png'),
            'pan': os.path.join(self.base_path, 'pan', file_name + '.jpg'),
        }

        try:
            with open(paths['annotation'], 'r') as f:
                annotation = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found for index {index}")

        rgb_image = read_image(paths['rgb'], cv2.IMREAD_COLOR)
        depth_image = read_image(paths['depth'], cv2.IMREAD_GRAYSCALE).unsqueeze(0)
        segmentation_image = read_image(paths['segmentation'], cv2.IMREAD_GRAYSCALE).unsqueeze(0)
        pan_image = read_image(paths['pan'], cv2.IMREAD_COLOR)

        return rgb_image, depth_image, segmentation_image, pan_image, annotation
