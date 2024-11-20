# Layers code for ICAMCS 2024 paper "Light-weight YOLOv8-seg Model Based on C2f-Faster-EMA For Ripeness Segmentation in Tomatoes and Applications"

Train CFE-Yolov8-seg in custom dataset tutorial
```bash
Step 1: Replace the task.py file of this repo to the original ultralytics repo.
Step 2: Place the CFE_module.py in the following direction "ultralytics/ultralytics/nn/modules".
Step 3: Use Yolo training CLI for our model.yaml for training and evaluations.
