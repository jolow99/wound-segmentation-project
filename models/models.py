
import sys 
import torch
import torchvision.models as models

sys.path.append('..')
from models.unet.modeling_simple_unet import SimpleUNet
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet

def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'simple_unet':
        config = UNetConfig(**config)
        model = SimpleUNet(config).to(device)
    elif model_name == 'mobilenet': 
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Replace the last layer with a new convolutional layer
        model.classifier = torch.nn.Conv2d(1280, 1, kernel_size=1)

        # Replace the last layer of the features module with a regular convolutional layer
        model.features[-1][0] = torch.nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        model.to(device)
        

        
    return model