import torch

from model import DroneDetector

from dataset import DroneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

model = DroneDetector().to(device)

model.load_state_dict(
    torch.load("drone_detector.pth", map_location=device)
)

model.eval()

print("Model loaded successfully!")


dataset = DroneDataset("val")

image, labels = dataset[0]
print("Ground truth labels:")
print(labels)

image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)

print("Output shape:", output.shape)


# Get predictions for the first image
prediction = output[0]

# Find the grid cell with the highest objectness
objectness = prediction[4]

max_value, max_index = torch.max(objectness.view(-1), dim=0)

grid_y = max_index // 40
grid_x = max_index % 40

print("Highest objectness:", max_value.item())
print("Grid X:", grid_x.item())
print("Grid Y:", grid_y.item())
print("Prediction at highest objectness cell:")
print(prediction[:, grid_y, grid_x])