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
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC to CHW for color images
    elif color_mode == cv2.IMREAD_GRAYSCALE:
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension for grayscale images

    return image

class MyDataset(torch.utils.data.Dataset):
    ANNOTATION_FOLDER = 'annotations'
    RGB_FOLDER = 'rgb'
    DEPTH_FOLDER = 'depth'
    SEGMENTATION_FOLDER = 'sem'
    # PAN_FOLDER = 'pan'

    def __init__(self, base_path: str):
        """

        :param base_path: The path to the dataset
        """
        self.base_path = base_path

        self.annotation_path = os.path.join(self.base_path, self.ANNOTATION_FOLDER)
        self.rgb_path = os.path.join(self.base_path, self.RGB_FOLDER)
        self.depth_path = os.path.join(self.base_path, self.DEPTH_FOLDER)
        self.segmentation_path = os.path.join(self.base_path, self.SEGMENTATION_FOLDER)
        # self.pan_path = os.path.join(self.base_path, self.PAN_FOLDER)

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
            
        # assert all(len(os.listdir(path)) == length for path in [self.rgb_path, self.depth_path, self.segmentation_path, self.pan_path]), "Inconsistent dataset sizes"
        # assert all(len(os.listdir(path)) == length for path in [self.rgb_path, self.depth_path, self.segmentation_path]), "Inconsistent dataset sizes"

        self.data_names = annotation_files_without_extension
        
    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index: int):
        file_name = self.data_names[index]
        paths = {
            'annotation': os.path.join(self.annotation_path, file_name + '.json'),
            'rgb': os.path.join(self.rgb_path, file_name + '-rgb.png'),
            'depth': os.path.join(self.depth_path, file_name + '-depth.png'),
            'segmentation': os.path.join(self.segmentation_path, file_name + '-sem.png'),
            # 'pan': os.path.join(self.pan_path, file_name + '-pan.png'),
        }

        try:
            with open(paths['annotation'], 'r') as f:
                annotation = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Annotation file not found for index {index}")

        rgb_image = read_image(paths['rgb'], cv2.IMREAD_COLOR)
        depth_image = read_image(paths['depth'], cv2.IMREAD_GRAYSCALE)
        segmentation_image = read_image(paths['segmentation'], cv2.IMREAD_GRAYSCALE)
        # pan_image = read_image(paths['pan'], cv2.IMREAD_COLOR)

        _data = {
            'rgb': rgb_image,
            'depth': depth_image,
            'sem': segmentation_image,
            # 'pan': pan_image,
            'annotation': annotation
        }
        
        return _data
    
def custom_collate_fn(batch):
    rgb = torch.stack([item['rgb'] for item in batch])
    depth = torch.stack([item['depth'] for item in batch])
    sem = torch.stack([item['sem'] for item in batch])
    # pan = torch.stack([item['pan'] for item in batch])
    annotation = [item['annotation'] for item in batch]  # Handle metadata separately
    
    return {
        'rgb': rgb,
        'depth': depth,
        'sem': sem,
        # 'pan': pan,
        'annotation': annotation
    }