import torch
import torch.nn as nn

class DroneDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # --- THE TRAP: Ensures this file is actually being loaded! ---
        print("\n SUCCESS: LOADING THE NEW BACKBONE ARCHITECTURE! \n")

        # Feature Extractor (Downsamples 8x via 3 MaxPool layers)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
            
            #nn.Conv2d(64, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)
        )

        # Decoupled Localization Head: outputs [tx, ty, tw, th] in [0, 1]
        # UPDATED: Input channels changed from 128 to 64
        self.bbox_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoupled Classification & Objectness Head: outputs raw logits [conf, class_1, ...]
        # UPDATED: Input channels changed from 128 to 64
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1 + num_classes, kernel_size=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        bbox_preds = self.bbox_head(features)
        cls_preds = self.cls_head(features)
        
        # Concatenate along channel dimension: [B, 4 + 1 + num_classes, H, W]
        return torch.cat([bbox_preds, cls_preds], dim=1)
