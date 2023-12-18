import sys, os
sys.path.append(os.path.join(os.getcwd(), "/scratch/xt2191/Grounded-Segment-Anything/GroundingDINO"))

from detectron2 import model_zoo
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg


from ultralytics import YOLO



class CondInstPredictor:
    def __init__(self, cfg):
        self.model_path = cfg.MODEL.WEIGHTS
        self.model = YOLO("/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/scripts/runs/detect/train3/weights/best.pt")

    def __call__(self, image):
        results = self.model(
            image[...,::-1],
        )

        return results


def load_maskrcnn(CLASS_TO_COLOR, SCREEN_SIZE):

    class_names = list(CLASS_TO_COLOR.keys())

    cfg = model_zoo.get_config("COCO-InstanceSegmentation/"
                               "mask_rcnn_R_50_FPN_3x.yaml")

    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)

    cfg.DATASETS.TEST = ('ai2thor-val',)
    cfg.DATASETS.TRAIN = ('ai2thor-train',)

    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    cfg.INPUT.MIN_SIZE_TRAIN = (SCREEN_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = SCREEN_SIZE

    cfg.INPUT.MIN_SIZE_TEST = SCREEN_SIZE
    cfg.INPUT.MAX_SIZE_TEST = SCREEN_SIZE

    cfg.TEST.AUG.MIN_SIZES = (SCREEN_SIZE,)
    cfg.TEST.AUG.MAX_SIZE = SCREEN_SIZE

    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "model_final.pth")

    return DefaultPredictor(cfg)


def load_sam_yolo(CLASS_TO_COLOR, SCREEN_SIZE):
    class_names = list(CLASS_TO_COLOR.keys())
    
    # Initialize the configuration
    cfg = get_cfg()
    
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.DATASETS.TEST = ('ai2thor-val',)
    cfg.DATASETS.TRAIN = ('ai2thor-train',)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TRAIN = (SCREEN_SIZE,)
    cfg.INPUT.MAX_SIZE_TRAIN = SCREEN_SIZE
    cfg.INPUT.MIN_SIZE_TEST = SCREEN_SIZE
    cfg.INPUT.MAX_SIZE_TEST = SCREEN_SIZE
    cfg.TEST.AUG.MIN_SIZES = (SCREEN_SIZE,)
    cfg.TEST.AUG.MAX_SIZE = SCREEN_SIZE


    cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/scripts/runs/detect/train3/weights/best.pt")
    return CondInstPredictor(cfg)


