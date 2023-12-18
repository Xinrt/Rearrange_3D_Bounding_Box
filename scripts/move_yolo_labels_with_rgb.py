import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def move_label_file(label_file):
    if label_file.endswith(".txt"):
        src_path = os.path.join(label_dir, label_file)
        dst_path = os.path.join(output_label_dir, label_file)
        shutil.move(src_path, dst_path)

if __name__ == "__main__":
    image_dir = "/vast/xt2191/dataset/rgb"
    label_dir = "/vast/xt2191/dataset/yolo_labels"
    output_label_dir = image_dir  # 将标签放在图像目录中

    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    max_workers = 20

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(move_label_file, label_files), total=len(label_files), desc="Moving Label Files"))

    print("标签文件已移动到图像目录")
