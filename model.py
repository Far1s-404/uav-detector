import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DroneDetector(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        print("\nSUCCESS: LOADING CUSTOM 8-CONV DRONE DETECTOR\n")

        
        # BACKBONE
        #
        # 1280 -> 640 -> 320 -> 160
        #
        # We deliberately DO NOT downsample below 160x160.
        # This preserves information about tiny drones.
        

        self.stage1 = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2, 2)
        )

        self.stage2 = nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2)
        )

        self.stage3 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2)
        )

        self.stage4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256)
        )

        
        # DETECTION HEAD
        #
        # Output:
        # 0: tx
        # 1: ty
        # 2: width
        # 3: height
        # 4: objectness
        # 5...: class probabilities
        #
        # Total = 4 + 1 + num_classes
        

        output_channels = 5 + num_classes

        self.head = nn.Sequential(
            ConvBlock(256, 256),
            nn.Conv2d(
                256,
                output_channels,
                kernel_size=1
            )
        )

    def forward(self, x):
        x = self.stage1(x)   # 1280 -> 640
        x = self.stage2(x)   # 640 -> 320
        x = self.stage3(x)   # 320 -> 160
        x = self.stage4(x)   # remains 160

        return self.head(x)
    
