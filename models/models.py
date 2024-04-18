
import sys 
sys.path.append('..')
from models.autoencoder.modeling_autoencoder import Autoencoder
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet
from models.circlenet.modeling_circlenet import CircleNet
from models.circlenet.configuration_circlenet import CircleNetConfig
from models.mixcirclenet.modeling_mixcirclenet import MixCircleNet
from models.mixcirclenet.configuration_mixcirclenet import MixCircleNetConfig
from models.deeplab.modeling_deeplabv3plus import DeepLabV3Plus
from models.segnet.modeling_segnet import SegNet
from models.fcn.modeling_fcn import FCN_Vgg16_16s
from models.segformer.configuration_segformer import SegformerConfig
from models.segformer.modeling_segformer import SegFormer
from models.pix2pix.pix2pix import Generator

#dont use _ for model names
def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'autoencoder':
        config = UNetConfig(**config)
        model = Autoencoder(config).to(device)
    elif model_name == 'circlenet':
        config = CircleNetConfig(**config)
        model = CircleNet(config).to(device)
    elif model_name == 'mixcirclenet':
        config = MixCircleNetConfig(**config)
        model = MixCircleNet(config).to(device)
    elif model_name == 'deeplabv3plus': 
        model = DeepLabV3Plus().to(device)
    elif model_name == 'segnet':
        model = SegNet().to(device)
    return model