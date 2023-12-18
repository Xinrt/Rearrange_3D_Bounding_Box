from ultralytics import YOLO

# Load a model
model = YOLO("/scratch/xt2191/mass/yolov8n.yaml").load("/scratch/xt2191/mass/yolov8n.pt")  
# build a new model from scratch and load weights from a file

# model = YOLO("/scratch/xt2191/mass/yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="/scratch/xt2191/mass/data/coco128.yaml", epochs=3)  # train the model
model.train(data="/scratch/xt2191/luyi/Rearrange_3D_Bounding_Box/Dataset/room.yaml", 
            imgsz=224, batch=256, workers=4,
            epochs=30, device=[0, 1])  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format