import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mass.thor.detectron_utils import load_maskrcnn, load_sam_yolo, load_ground
from mass.thor.segmentation_config import CLASS_TO_COLOR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from grounded_fast_sam_modified import run_grounding_fast_sam, parse_args
from grounded_sam_demo_modified import run_SAM
import csv

from Levenshtein import ratio


# Configuration
detection_threshold = 0.0
NUM_CLASSES = len(CLASS_TO_COLOR)
logdir = "./test_maskrcnn_saveimage"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
sem_seg_model = load_maskrcnn(CLASS_TO_COLOR, 224)
sem_seg_model.model.to(device)

# sem_seg_model_new = load_sam_yolo(CLASS_TO_COLOR, 224)
# sem_seg_model_new.model.to(device)

sem_seg_model_new = load_ground(CLASS_TO_COLOR, 224)
sem_seg_model_new.model.to(device)


# # use own model
# sem_seg_model_new = load_yolo(CLASS_TO_COLOR, 224)



def segements(frame):
    outputs = sem_seg_model(frame[:, :, ::-1])
    semantic_seg = torch.zeros(224, 224,
                                    NUM_CLASSES,
                                    device=device,
                                    dtype=torch.float32)
    # print(outputs['instances'].scores)
    # print(outputs['instances'].pred_classes)
    # print("outputs['instances']:\n", outputs['instances'])

    for i in range(len(outputs['instances'])):        
        object_score = outputs['instances'].scores[i]

        if object_score < detection_threshold:
            continue  # skip if the model is not confident enough

        object_class = outputs['instances'].pred_classes[i]
        # print("mask mask sobject_class:\n", object_class)

        # otherwise, add the object mask to the segmentation buffer
        semantic_seg[:, :, object_class] += \
            outputs['instances'].pred_masks[i].to(torch.float32)

    # take argmax over the channels to identify one object per pixel
    # semantic_seg = semantic_seg.argmax(dim=2, keepdim=True)
    # take argmax over the channels to identify one object per pixel
    # for i in range(NUM_CLASSES):
    #     print(f"mask  mask  Non-zero values in channel {i}:", (semantic_seg[:,:,i] > 0).sum().item())

    # print("Non-zero values in semantic_seg before 2:", semantic_seg.nonzero().size(0))
    semantic_seg = semantic_seg.mean(dim=2, keepdim=True)
    # print("Non-zero values in semantic_seg after 2:", semantic_seg.nonzero().size(0))
    # import pdb; pdb.set_trace()
    return semantic_seg

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
    

# def get_name(frame, img_path):
#     print("CLASS_TO_COLOR:\n", CLASS_TO_COLOR)
#     class_name = "Unknown"
#     # Initialize an empty list to store the filtered bounding boxes
#     boxes_filt_list = []
#     class_name_list = []

#     # Get the YOLO model's output
#     outputs = sem_seg_model_new(frame[:, :, ::-1])

#     # print("outputs:\n", outputs)
#     # Initialize the semantic segmentation buffer
#     # semantic_seg = torch.zeros(224, 224, NUM_CLASSES, device=device, dtype=torch.float32)
#     semantic_seg = torch.zeros(224, 224, len(outputs[0].names), device=device, dtype=torch.float32)

#     results_object = outputs[0]
#     # print("outputs[0]:\n", outputs[0])
#     # print("outputs[0].boxes:\n", outputs[0].boxes)
#     frame_with_boxes = frame.copy()  # Create a copy of the original image to draw boxes

#     # Extract image, names, boxes from the result
#     img = outputs[0].orig_img
#     names = outputs[0].names

#     # Assuming boxes object has xyxy and scores for this example
#     # Transfer to CPU and convert to numpy
#     boxes_xyxy = outputs[0].boxes.xyxy.cpu().numpy()
#     scores = outputs[0].boxes.conf.cpu().numpy()

#     print("outputs[0].boxes.xyxy: ", outputs[0].boxes.xyxy)
#     print("outputs[0].boxes.conf: ", outputs[0].boxes.conf)

#     # Plotting the original image
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img)
#     plt.axis('off')  # to hide axis labels

#     # Draw each bounding box
#     for box, score in zip(boxes_xyxy, scores):
#         # Extract coordinates
#         x1, y1, x2, y2 = box
#         width = x2 - x1
#         height = y2 - y1
        
#         # Create a rectangle patch
#         rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
#         plt.gca().add_patch(rect)

#             # Plot confidence score
#         plt.text(x1, y1 - 5, f"{score:.2%}", color='red', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=0.7))

#     plt.show()

#     import pdb; pdb.set_trace();

#     # Extract image, names, boxes from the result
#     img = outputs[0].orig_img
#     names = outputs[0].names

#     # Transfer to CPU and convert to numpy
#     boxes_xyxy = outputs[0].boxes.xyxy.cpu().numpy()
#     scores = outputs[0].boxes.conf.cpu().numpy()

#     # Filter out boxes based on detection threshold
#     filtered_boxes = []
#     filtered_scores = []
#     filtered_class_names = []


#     print("len(outputs[0].boxes): ", len(outputs[0].boxes))
#     for i in range(len(outputs[0].boxes)):
#         if scores[i] < detection_threshold:
#             continue


#         # import pdb; pdb.set_trace();


#         # # print("outputs[0].boxes.cls:\n", outputs[0].boxes.cls)
#         # object_class = outputs[0].boxes.cls[i].long()
#         # # print("object_class:\n", object_class)
#         # # print("outputs[0].names[object_class.item()]:", outputs[0].names[object_class.item()])

#         # # 把所有能检测到的物体拿出来，看是不是在54里面，大于0.3的
#         # class_name = outputs[0].names[object_class.item()]

#         # class_name_list.append(outputs[0].names[object_class.item()])
#         # boxes_filt_list.append(outputs[0].boxes.xywhn)

#         object_class = outputs[0].boxes.cls[i].long()
#         class_name = outputs[0].names[object_class.item()]

#         filtered_boxes.append(boxes_xyxy[i])
#         filtered_scores.append(scores[i])
#         filtered_class_names.append(class_name)

#         # class_name_list.append(class_name)
#         # boxes_filt_list.append(outputs[0].boxes.xywhn)
#         class_name_list.append(class_name)
#         boxes_filt_list.append(outputs[0].boxes.xywhn[i])

#     print()

#     # # Plotting the original image
#     # plt.figure(figsize=(12, 12))
#     # plt.imshow(img)
#     # plt.axis('off')  # to hide axis labels

#     # # Draw each filtered bounding box
#     # for box, score, class_name in zip(filtered_boxes, filtered_scores, filtered_class_names):
#     #     # Extract coordinates
#     #     x1, y1, x2, y2 = box
#     #     width = x2 - x1
#     #     height = y2 - y1

#     #     # Create a rectangle patch
#     #     rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
#     #     plt.gca().add_patch(rect)

#     #     # Plot confidence score and class name
#     #     plt.text(x1, y1 - 5, f"{class_name}: {score:.2%}", color='red', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1', alpha=0.7))

#     # plt.show()
#     # import pdb; pdb.set_trace();
        
    
#     # print("boxes_filt_list:\n", boxes_filt_list)

#     # Convert the list of bounding boxes to a PyTorch tensor
#     print("len(boxes_filt_list): ", len(boxes_filt_list))
#     print("boxes_filt_list:\n", boxes_filt_list)
#     print("class_name_list:\n", class_name_list)

#     if len(boxes_filt_list) == 0:
#         mask_sam = torch.zeros(1, 1, 224, 224, device='cuda:0')
#     else:
#         boxes_filt = boxes_filt_list
#         mask_sam = run_SAM(img_path, boxes_filt, class_name_list)


#     # import pdb; pdb.set_trace();

#     return class_name, mask_sam



def get_name(frame, img_path):
    class_name = "Unknown"
    # Initialize an empty list to store the filtered bounding boxes
    boxes_filt_list = []
    class_name_list = []

    # Get the YOLO model's output
    outputs = sem_seg_model_new(frame[:, :, ::-1])
    semantic_seg = torch.zeros(224, 224, len(outputs[0].names), device=device, dtype=torch.float32)

    results_object = outputs[0]
    print("results_object:\n", outputs[0])
    print("CLASS_TO_COLOR:\n", CLASS_TO_COLOR)

    # import pdb; pdb.set_trace();

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
        # class_name = outputs[0].names[object_class.item()]
        detected_name = outputs[0].names[object_class.item()]
        print("detected_name: ", detected_name)

        # Map the detected name to CLASS_TO_COLOR key based on similarity
        mapped_name = find_similar_key(detected_name, CLASS_TO_COLOR.keys())

        # If the detected name has a similar key in CLASS_TO_COLOR or directly exists in CLASS_TO_COLOR
        if mapped_name or detected_name in CLASS_TO_COLOR:
            filtered_boxes.append(boxes_xyxy[i])
            filtered_scores.append(scores[i])
            filtered_class_names.append(detected_name) 

            class_name_list.append(detected_name)
            boxes_filt_list.append(outputs[0].boxes.xywhn[i])




        



    # Convert the list of bounding boxes to a PyTorch tensor
    print("len(boxes_filt_list): ", len(boxes_filt_list))
    print("boxes_filt_list:\n", boxes_filt_list)
    print("class_name_list:\n", class_name_list)

    if len(boxes_filt_list) == 0:
        mask_sam = torch.zeros(1, 1, 224, 224, device='cuda:0')
    else:
        boxes_filt = boxes_filt_list
        mask_sam = run_SAM(img_path, boxes_filt, class_name_list)



    return class_name, mask_sam

def segements_new(frame, img_path):
    class_name, mask = get_name(frame, img_path)


    # # 创建一个argparse的Namespace对象并填充参数
    # args = parse_args()
    # args.model_path = "/scratch/xt2191/mass/mass/thor/FastSAM-x.pt"
    # args.img_path = img_path
    # args.text = class_name
    # args.output = "/scratch/xt2191/mass/seg_output"

    # try:
    #     semantic_seg, img_array, mask = run_grounding_fast_sam(args)
    # except:
    #     print("Error processing image:", img_path)
    #     resized_frame = cv2.resize(frame, (224, 224))
    #     mask = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)





    # # Display the mask using plt
    # plt.imshow(mask)  # Assuming mask is grayscale
    # plt.title('Detected Mask')
    # plt.axis('off')
    # plt.show()




    # import pdb; pdb.set_trace();

    return mask

def calculate_metrics(pred, gts):
    """Calculate precision, recall, and F1 score."""
    TP = ((pred == 1) & (gts == 1)).sum()
    FP = ((pred == 1) & (gts == 0)).sum()
    FN = ((pred == 0) & (gts == 1)).sum()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1



def cust_accuracy(input, targs):
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    # input = np.argmax(input, axis=2)
    input = input.reshape(1,-1)
    targs = targs.reshape(1,-1)
    return (input==targs).mean()

def jaccard_score(pred, gts):
    jaccard = []
    for i in range(54):
        intersection = ((pred==i)*(gts==i)).sum()
        union = (pred==i).sum()+(gts==i).sum()-intersection
        jaccard.append(float(intersection)/(union+1e-8))
    return sum(jaccard) / len(jaccard)

# Process images
accuracy = []
iou = []
precision = []
recall = []
f1 = []

accuracy_new = []
iou_new = []
precision_new = []
recall_new = []
f1_new = []
# writer = SummaryWriter(logdir)

# start_image_id = 17500
# end_image_id = start_image_id + 20

start_image_id = 24936
end_image_id = start_image_id + 1

for image_id in range(start_image_id, end_image_id):
# for image_id in range(100):
    image_id_str = f"{image_id:07d}"
    print("image_id", image_id)
    rgb_path = os.path.join("/scratch/xt2191/mass/test-data/images", f"{image_id_str}-rgb.png")
    sem_path = os.path.join("/scratch/xt2191/mass/test-data/images", f"{image_id_str}-sem.png")

    frame = cv2.imread(rgb_path)
    GT_seg = cv2.imread(sem_path)[:,:,1:2]
    # GT_seg = cv2.imread(sem_path)

    frame_resized = cv2.resize(frame, (224, 224))
    predict_seg = segements(frame_resized)
    predict_seg_new = segements_new(frame_resized, rgb_path)
    # print("predict_seg_new:\n", predict_seg_new.unique())


    js = jaccard_score(predict_seg.cpu().numpy(), GT_seg)
    iou.append(js)
    acc = cust_accuracy(predict_seg.cpu().numpy(), GT_seg)
    accuracy.append(acc)

    precision_0, recall_0, f1_0 = calculate_metrics(predict_seg.cpu().numpy(), GT_seg)
    precision.append(precision_0)
    recall.append(recall_0)
    f1.append(f1_0)



    # predict_seg_new = predict_seg_new[:, :, np.newaxis]
    # predict_seg_new = predict_seg_new.squeeze(0).squeeze(0).unsqueeze(-1)

    # print("predict_seg_new.shape:\n", predict_seg_new.shape)    
    # print("GT_seg.shape:\n", GT_seg.shape)

    # predict_seg_new = predict_seg_new[0, 0, :, :].unsqueeze(-1)
    predict_seg_new = predict_seg_new.squeeze(0).unsqueeze(-1)




    print("predict_seg_new.shape:\n", predict_seg_new.shape)
    print("GT_seg.shape:\n", GT_seg.shape)
    # predict_seg_new = predict_seg_new[:,:,1:2]
    # js_new = jaccard_score(predict_seg_new, GT_seg)
    # iou_new.append(js_new)
    # acc_new = cust_accuracy(predict_seg_new, GT_seg)
    # accuracy_new.append(acc_new)

    js_new = jaccard_score(predict_seg_new.cpu().numpy(), GT_seg)
    iou_new.append(js_new)
    acc_new = cust_accuracy(predict_seg_new.cpu().numpy(), GT_seg)
    accuracy_new.append(acc_new)

    precision_new_0, recall_new_0, f1_new_0 = calculate_metrics(predict_seg_new.cpu().numpy(), GT_seg)
    precision_new.append(precision_new_0)
    recall_new.append(recall_new_0)
    f1_new.append(f1_new_0)





    fig, _axs = plt.subplots(nrows=2, ncols=2, figsize=(30, 5))
    axs = _axs.flatten()
    axs[0].imshow(frame)
    axs[0].axis('off')
    axs[0].set_title('Original Frame')
    axs[1].imshow(GT_seg)
    axs[1].axis('off')
    axs[1].set_title('Ground Truth Segmentation')
    axs[2].imshow(predict_seg.cpu().numpy())
    axs[2].axis('off')
    axs[2].set_title('Mask RCNN')
    axs[3].imshow(predict_seg_new.cpu().numpy().squeeze())
    axs[3].axis('off')
    axs[3].set_title('SAM - YOLO')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(logdir, f"{image_id_str}-visual.png"))
    plt.close()

    # writer.add_scalars("log", {'jaccard_score': js, 'accuracy': acc}, image_id)



# Define the CSV file path
csv_file_path = "/scratch/xt2191/mass/test_maskrcnn_saveimage/results.csv"

# Check if the CSV file exists. If it doesn't, write the header.
try:
    with open(csv_file_path, 'r') as f:
        pass
except FileNotFoundError:
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Metric", "Value"])

# Write the values to the CSV
with open(csv_file_path, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["accuracy", accuracy])
    csvwriter.writerow(["iou", iou])
    csvwriter.writerow(["precision", precision])
    csvwriter.writerow(["recall", recall])
    csvwriter.writerow(["f1", f1])

    csvwriter.writerow(["accuracy_new", accuracy_new])
    csvwriter.writerow(["iou_new", iou_new])
    csvwriter.writerow(["precision_new", precision_new])
    csvwriter.writerow(["recall_new", recall_new])
    csvwriter.writerow(["f1_new", f1_new])

print("Values saved to", csv_file_path)

print("accuracy: ", accuracy)
print("iou: ", iou)
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", f1)

print("accuracy_new: ", accuracy_new)
print("iou_new: ", iou_new)
print("precision_new: ", precision_new)
print("recall_new: ", recall_new)
print("f1_new: ", f1_new)

# Define the directory to save the plots
save_dir = "/scratch/xt2191/mass/test_maskrcnn_saveimage/"

metrics = ["accuracy", "iou", "precision", "recall", "f1"]
old_values = [accuracy, iou, precision, recall, f1]
new_values = [accuracy_new, iou_new, precision_new, recall_new, f1_new]

for metric, old, new in zip(metrics, old_values, new_values):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(old)), old, label=f"Old {metric}", color="blue", marker='o')
    plt.plot(range(len(new)), new, label=f"New {metric}", color="red", marker='x')
    plt.xlabel('Image Index')
    plt.ylabel(metric)
    plt.title(f"{metric} Comparison between Old and New Implementation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the figure in the specified directory with the metric name as filename
    plt.savefig(os.path.join(save_dir, f"{metric}_comparison.png"))
    
    # plt.show()
