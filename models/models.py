
import sys 
sys.path.append('..')
from models.unet.modeling_simple_unet import SimpleUNet
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet
from models.deeplab.modeling_deeplabv3plus import DeepLabV3Plus

def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'simple_unet':
        config = UNetConfig(**config)
        model = SimpleUNet(config).to(device)
    elif model_name == 'deeplabv3plus': 
        model = DeepLabV3Plus().to(device)
    return model