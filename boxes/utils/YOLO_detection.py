import torch
from mass.thor.detectron_utils import load_sam_yolo
from mass.thor.segmentation_config import CLASS_TO_COLOR
import torch.nn.functional as F

from Levenshtein import ratio


# Configuration
detection_threshold = 0.0
NUM_CLASSES = len(CLASS_TO_COLOR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
sem_seg_model_new = load_sam_yolo(CLASS_TO_COLOR, 224)
sem_seg_model_new.model.to(device)



def find_similar_key(detected_name, class_to_color_keys, similarity_threshold=0.8):
    """
    Find a similar key in class_to_color_keys based on Levenshtein distance.
    """
    max_similarity = 0
    best_match = None
    
    for key in class_to_color_keys:
        similarity = ratio(detected_name.lower(), key.lower())

        if similarity > max_similarity:
            max_similarity = similarity
            best_match = key
            
    if max_similarity > similarity_threshold:
        return best_match

    return None


def generate_2dboxes(frame):
    class_name = "Unknown"
    # Initialize an empty list to store the filtered bounding boxes
    boxes_filt_list = []
    class_name_list = []

    # Get the YOLO model's output
    outputs = sem_seg_model_new(frame[:, :, ::-1])
    semantic_seg = torch.zeros(224, 224, len(outputs[0].names), device=device, dtype=torch.float32)

    results_object = outputs[0]
    # print("results_object:\n", outputs[0])
    # print("CLASS_TO_COLOR:\n", CLASS_TO_COLOR)

    frame_with_boxes = frame.copy()  # Create a copy of the original image to draw boxes


    # Extract image, names, boxes from the result
    img = outputs[0].orig_img
    names = outputs[0].names

    # Transfer to CPU and convert to numpy
    boxes_xyxy = outputs[0].boxes.xyxy.cpu().numpy()
    scores = outputs[0].boxes.conf.cpu().numpy()

    # Filter out boxes based on detection threshold
    filtered_boxes = []
    filtered_scores = []
    filtered_class_names = []


    for i in range(len(outputs[0].boxes)):
        if scores[i] < detection_threshold:
            continue

        object_class = outputs[0].boxes.cls[i].long()
        detected_name = outputs[0].names[object_class.item()]
        # print("detected_name: ", detected_name)

        # Map the detected name to CLASS_TO_COLOR key based on similarity
        mapped_name = find_similar_key(detected_name, CLASS_TO_COLOR.keys())

        # If the detected name has a similar key in CLASS_TO_COLOR or directly exists in CLASS_TO_COLOR
        if mapped_name or detected_name in CLASS_TO_COLOR:
            filtered_boxes.append(boxes_xyxy[i])
            filtered_scores.append(scores[i])
            filtered_class_names.append(detected_name) 

            class_name_list.append(detected_name)
            boxes_filt_list.append(outputs[0].boxes.xywhn[i])





    return filtered_boxes, class_name_list


