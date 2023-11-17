import os
import cv2
from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Constants for CLASS_TO_COLOR
# Modify these if necessary
CLASS_TO_COLOR = {
    "Candle": (255, 0, 0)
    # Add more class to color mappings as needed
}

def load_maskrcnn():
    class_names = list(CLASS_TO_COLOR.keys())
    cfg = model_zoo.get_config("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return cfg

def draw_cube(img, bbox):
    # Draw a cube around the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    offset = int(h / 4)
    
    # Define cube vertices
    vertices = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
        (x1+offset, y1-offset), (x2+offset, y1-offset), (x2+offset, y2-offset), (x1+offset, y2-offset)
    ]
    
    # Define connections between vertices for the cube
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom square
        (4,5), (5,6), (6,7), (7,4),  # Top square
        (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    ]

    for start, end in edges:
        cv2.line(img, vertices[start], vertices[end], (0, 255, 0), 2)

def main(args):
    cfg = load_maskrcnn()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 54
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = args.model_path

    predictor = DefaultPredictor(cfg)
    im = cv2.imread(args.img_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    # Draw instance masks and bounding boxes
    masks = instances.pred_masks
    bboxes = instances.pred_boxes.tensor
    for mask, bbox in zip(masks, bboxes):
        # im[mask] = 0.5 * im[mask] + 0.5 * CLASS_TO_COLOR['Candle']  # Change 'Candle' if you have different class names.
        # for c in range(3):  # for each channel: R, G, B
        #     im[mask, c] = 0.5 * im[mask, c] + 0.5 * CLASS_TO_COLOR['Candle'][c]

        draw_cube(im, bbox)

    cv2.putText(im, args.text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    output_path = os.path.join(args.output, os.path.basename(args.img_path))
    cv2.imwrite(output_path, im)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Mask R-CNN instance segmentation with Detectron2")
    parser.add_argument("--model_path", required=True, help="Path to the model weights")
    parser.add_argument("--img_path", required=True, help="Path to the input image")
    parser.add_argument("--text", type=str, default="", help="Text label to overlay on the image")
    parser.add_argument("--output", required=True, help="Path to save the segmented image")

    args = parser.parse_args()
    main(args)
