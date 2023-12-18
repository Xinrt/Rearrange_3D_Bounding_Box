import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def coco_to_yolo(bbox, img_width, img_height):
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w /= img_width
    h /= img_height
    return [x_center, y_center, w, h]

def process_annotation(file_path, output_dir):
    with open(file_path, 'r') as file:
        data = json.load(file)

    img_name = data["file_name"]
    img_width = data["width"]
    img_height = data["height"]
    annotations = data["annotations"]

    txt_name = os.path.splitext(os.path.basename(img_name))[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    with open(txt_path, 'w') as txt_file:
        for annotation in annotations:
            category_id = annotation["category_id"]
            bbox = annotation["bbox"]
            yolo_bbox = coco_to_yolo(bbox, img_width, img_height)
            line = f"{category_id} {' '.join(map(str, yolo_bbox))}\n"
            txt_file.write(line)

def main():
    annotation_dir = "/vast/xt2191/dataset/annotations"
    output_dir = "/vast/xt2191/dataset/yolo_labels"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotation_files = glob.glob(os.path.join(annotation_dir, "*.json"))

    # 设置合适的线程池大小
    pool_size = 15

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        # 使用tqdm显示进度
        list(tqdm(executor.map(lambda file: process_annotation(file, output_dir), annotation_files), total=len(annotation_files)))

    print("Conversion to YOLO format completed!")

if __name__ == "__main__":
    main()
