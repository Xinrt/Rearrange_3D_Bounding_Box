import os

def clean_labels(directory, max_class_id=53):
    for label_file in os.listdir(directory):
        if label_file.endswith(".txt"):
            filepath = os.path.join(directory, label_file)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            valid_lines = []
            for line in lines:
                class_id = int(line.split()[0])
                if class_id <= max_class_id:
                    valid_lines.append(line)
            
            with open(filepath, 'w') as f:
                f.writelines(valid_lines)

label_dir = "/scratch/xt2191/FastSAM/datasets/coco128/labels/train2017"
clean_labels(label_dir)
