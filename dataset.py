from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

DATASET_PATH = Path("D:/uav_detector/archive")

IMAGE_PATH = DATASET_PATH / "images" / "train"
LABEL_PATH = DATASET_PATH / "labels" / "train"

IMAGE_SIZE = 640

def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    images = torch.stack(images)

    return images, labels

class DroneDataset(Dataset):

    def __init__(self, split="train"):

        image_path = DATASET_PATH / "images" / split
        label_path = DATASET_PATH / "labels" / split

        self.images = list(image_path.glob("*.jpg"))
        self.label_path = label_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image_path = self.images[index]

        # Load image && resize (bn3ml resize to image and boundary box too )
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = transforms.ToTensor()(image)


        # Find corresponding label
        label_path = self.label_path / (image_path.stem + ".txt")

        labels = []

        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()

                class_id = int(values[0])
                x = float(values[1])
                y = float(values[2])
                width = float(values[3])
                height = float(values[4])

                labels.append([
                    class_id,
                    x,
                    y,
                    width,
                    height
                ])

        return image, torch.tensor(labels, dtype=torch.float32)


dataset = DroneDataset()

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)
images, labels = next(iter(dataloader))
print("Batch image shape:", images.shape)
print("Number of labels:", len(labels))
print("Number of images:", len(dataset))

image, labels = dataset[0]

print("Image tensor shape:", image.shape)
print("Labels:")
print(labels)