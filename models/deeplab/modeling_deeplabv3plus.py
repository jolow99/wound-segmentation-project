import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        self.convs = nn.ModuleList()

        # 1x1 conv
        self.convs.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        # Three 3x3 convs with different atrous_rates
        for rate in atrous_rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        # Image pooling
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        image_features = F.interpolate(self.image_pooling(x), size=size, mode='bilinear', align_corners=False)

        aspp_features = [conv(x) for conv in self.convs]
        aspp_features.append(image_features)
        return torch.cat(aspp_features, dim=1)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP(in_channels, [12, 24, 36])  # Atrous rates for an output stride of 16
        aspp_out_channels = 5 * 256  # 5 branches in ASPP, each with 256 channels
        self.conv = nn.Conv2d(aspp_out_channels, 256, 1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.last_conv = nn.Conv2d(256, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.aspp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.last_conv(x)
        return x



# Assuming we are using output stride 16, we do not modify the backbone
# However for output stride 8, we would need to apply modifications
# to the backbone to have dilated convolutions in the last block

# Construct the DeepLabV3+ model
class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()
        
        # Load a pretrained MobileNetV2 model
        self.backbone = models.mobilenet_v2(pretrained=True).features

        # Only 1 output class
        self.head = DeepLabHead(1280, 1)  # 1280 is the number of output channels from MobileNetV2. 

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
