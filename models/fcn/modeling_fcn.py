import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN_Vgg16_16s(nn.Module):
    def __init__(self, num_classes=1, weight_decay=0., batch_momentum=0.9):
        super(FCN_Vgg16_16s, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.block1_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 2
        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.block2_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 3
        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.block3_conv3 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.block3_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 4
        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.block4_conv3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.block4_pool = nn.MaxPool2d(2, stride=2)
        
        # Block 5
        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        self.block5_conv3 = nn.Conv2d(512, 512, 3, padding=1, bias=False)
        
        # Convolutional layers transferred from fully-connected layers
        self.fc1 = nn.Conv2d(512, 4096, 7, padding=3, dilation=2, bias=False)
        self.fc2 = nn.Conv2d(4096, 4096, 1, bias=False)
        
        # Classifying layer
        self.classifier = nn.Conv2d(4096, num_classes, 1, bias=False)
        
        # Upsampling layer
        # self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=16, bias=False)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.block1_conv1(x))
        x = F.relu(self.block1_conv2(x))
        x = self.block1_pool(x)  # Output size: 112x112

        # Block 2
        x = F.relu(self.block2_conv1(x))
        x = F.relu(self.block2_conv2(x))
        x = self.block2_pool(x)  # Output size: 56x56

        # Block 3
        x = F.relu(self.block3_conv1(x))
        x = F.relu(self.block3_conv2(x))
        x = F.relu(self.block3_conv3(x))
        x = self.block3_pool(x)  # Output size: 28x28

        # Block 4
        x = F.relu(self.block4_conv1(x))
        x = F.relu(self.block4_conv2(x))
        x = F.relu(self.block4_conv3(x))
        x = self.block4_pool(x)  # Output size: 14x14

        # Block 5
        x = F.relu(self.block5_conv1(x))
        x = F.relu(self.block5_conv2(x))
        x = F.relu(self.block5_conv3(x))  # Output size: 14x14

        # Convolutional layers transferred from fully-connected layers
        x = F.relu(self.fc1(x))  # Output size: 7x7
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))  # Output size: 7x7
        x = F.dropout(x, 0.5)

        # Classifying layer
        x = self.classifier(x)  # Output size: 7x7

        # Upsampling layer
        # x = self.upsample(x) 
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        # Output size: 224x224

        return x