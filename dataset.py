from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_PATH = Path("D:/uav_detector/archive")
IMAGE_SIZE = 640

def collate_fn(batch):
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    images = torch.stack(images, dim=0)
    return images, labels

class DroneDataset(Dataset):
    def __init__(self, split="train"):
        self.image_dir = DATASET_PATH / "images" / split
        self.label_dir = DATASET_PATH / "labels" / split
        self.split = split

        # --- THE SPLIT AUGMENTATION PIPELINE ---
        if self.split == "train":
            # Training: Aggressive pixel distortion to prevent overfitting
            self.transforms = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.5, hue=0.2),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), 
                transforms.ToTensor(),
            ])
        else:
            # Validation: Untouched and clean for honest scoring
            self.transforms = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ])

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.images = [
            p for p in self.label_dir.parent.parent.glob(f"images/{split}/*")
            if p.suffix.lower() in valid_extensions
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = Image.open(image_path).convert("RGB")
        
        # Apply the correct transforms based on the split
        image_tensor = self.transforms(image)

        label_file = self.label_dir / f"{image_path.stem}.txt"
        labels = []

        if label_file.exists():
            with open(label_file, "r") as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) >= 5:
                        labels.append([
                            float(values[0]),  
                            float(values[1]),  
                            float(values[2]),  
                            float(values[3]),  
                            float(values[4])   
                        ])

        if len(labels) == 0:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return image_tensor, labels_tensor
