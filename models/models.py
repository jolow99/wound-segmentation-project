
import sys 
from torchvision import models

sys.path.append('..')
from models.unet.modeling_simple_unet import SimpleUNet
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet
from models.deeplab.modeling_deeplabv3plus import DeepLabV3Plus, DeepLabHead

def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'simple_unet':
        config = UNetConfig(**config)
        model = SimpleUNet(config).to(device)
    elif model_name == 'deeplabv3plus': 
        # Load a pretrained MobileNetV2 model
        model = models.mobilenet_v2(pretrained=True)

        # Modify the MobileNetV2 model to replace the classifier with the DeepLab head
        backbone = model.features

        # Create the DeepLab head with the number of classes, for example 21 for VOC
        num_classes = 1
        deeplab_head = DeepLabHead(1280, num_classes)  # 1280 is the number of output channels from MobileNetV2

        model = DeepLabV3Plus(backbone, deeplab_head)
        model = model.to(device)
    return model