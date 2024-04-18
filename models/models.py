
import sys 
sys.path.append('..')
from models.unet.modeling_simple_unet import SimpleUNet
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet
from models.deeplab.modeling_deeplabv3plus import DeepLabV3Plus
from models.segnet.modeling_segnet import SegNet
from models.pix2pix.pix2pix import Generator

#dont use _ for model names
def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'simpleunet':
        config = UNetConfig(**config)
        model = SimpleUNet(config).to(device)
    elif model_name == 'deeplabv3plus': 
        model = DeepLabV3Plus().to(device)
    elif model_name == 'segnet':
        model = SegNet().to(device)
    elif model_name == 'pix2pix':
        model = Generator().to(device)
    else: 
        raise ValueError(f"Model {model_name} not found")
    return model