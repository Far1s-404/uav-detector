import torch
import cv2
import torchvision.ops as ops
from pathlib import Path
from model import DroneDetector
from dataset import DroneDataset

device = torch.device("cpu")
print(" Using device:", device)

# --- CONFIGURATION ---
# Fixed the path to match where train.py is saving the weights
weights_path = Path("D:/uav_detector/weights/best_drone_detector.pth")
CONFIDENCE_THRESHOLD = 0.01 
IOU_THRESHOLD = 0.3  # Controls how aggressively to delete overlapping boxes

# 1. Initialize model with 5 drone classes and load weights
model = DroneDetector(num_classes=5).to(device)

if not weights_path.exists():
    print(f" Error: Weights file not found at {weights_path}")
    exit()

model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()
print(" Model loaded successfully!")

# 2. Load evaluation sample
dataset = DroneDataset("val")
sample_index =16

image, labels = dataset[sample_index]
input_tensor = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)

prediction = output[0]
_, grid_h, grid_w = prediction.shape

# 3. Apply Sigmoid to get 0-1 confidence percentages
obj_conf = torch.sigmoid(prediction[4])

# 4. Find ALL grid cells that beat the threshold
mask = obj_conf > CONFIDENCE_THRESHOLD
ys, xs = torch.where(mask)

# Load original raw image to fetch native resolutions for OpenCV
image_path = Path("D:/uav_detector/archive/images/val") / dataset.images[sample_index].name
original_image = cv2.imread(str(image_path))
orig_h, orig_w = original_image.shape[:2]

boxes = []
scores = []
class_ids = []

# Decode every cell that passed the threshold
for grid_y, grid_x in zip(ys, xs):
    confidence = obj_conf[grid_y, grid_x].item()
    
    cell_x = prediction[0, grid_y, grid_x].item()
    cell_y = prediction[1, grid_y, grid_x].item()
    norm_w = prediction[2, grid_y, grid_x].item()
    norm_h = prediction[3, grid_y, grid_x].item()

    norm_cx = (grid_x.item() + cell_x) / grid_w
    norm_cy = (grid_y.item() + cell_y) / grid_h

    center_x = int(norm_cx * orig_w)
    center_y = int(norm_cy * orig_h)
    box_w = int(norm_w * orig_w)
    box_h = int(norm_h * orig_h)

    x1 = max(0, center_x - box_w // 2)
    y1 = max(0, center_y - box_h // 2)
    x2 = min(orig_w - 1, center_x + box_w // 2)
    y2 = min(orig_h - 1, center_y + box_h // 2)

    boxes.append([x1, y1, x2, y2])
    scores.append(confidence)
    
    # Extract class
    class_scores = prediction[5:10, grid_y, grid_x]
    class_ids.append(torch.argmax(class_scores).item())

class_names = [
    "shahed_136", "shahed_238", "mq9_reaper", "dji_mavic", "mohajer_6"
]

# 5. Apply Non-Maximum Suppression (NMS) and Draw
if len(boxes) > 0:
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    # NMS filters out multiple boxes detecting the exact same drone
    keep_indices = ops.nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)
    
    print(f"🎯 Found {len(keep_indices)} drone(s)!")
    
    for idx in keep_indices:
        x1, y1, x2, y2 = boxes[idx]
        conf = scores[idx]
        cls_id = class_ids[idx]
        drone_label = class_names[cls_id]
        
        print(f" -> {drone_label} | Confidence: {conf:.4f} | Box: ({x1}, {y1}) to ({x2}, {y2})")

        # Render annotations
        cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        tag = f"{drone_label} {conf:.2f}"
        cv2.putText(
            original_image, tag, (int(x1), max(int(y1) - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
else:
    print(" No drones detected above the confidence threshold.")

output_path = Path("D:/uav_detector/deep_cv_model/custom_detection.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output_path), original_image)
print(f" Detection image successfully saved to: {output_path}")
