
import sys 
sys.path.append('..')
from models.autoencoder.modeling_autoencoder import Autoencoder
from models.unet.configuration_unet import UNetConfig
from models.unet.modeling_unet import UNet
from models.unet.modeling_unet_ensemble import EnsembleUNet
from models.unet.modeling_unet_crf import UNetCRF
from models.deeplab.modeling_deeplabv3plus import DeepLabV3Plus
from models.segnet.modeling_segnet import SegNet
from models.fcn.modeling_fcn import FCN_Vgg16_16s
from models.segformer.configuration_segformer import SegformerConfig
from models.segformer.modeling_segformer import SegFormer

#dont use _ for model names
def get_model(model_name, config, device="cuda"): 
    if model_name == 'unet': 
        config = UNetConfig(**config)
        model = UNet(config).to(device)
    elif model_name == 'autoencoder':
        config = UNetConfig(**config)
        model = Autoencoder(config).to(device)
    elif model_name == 'unetensemble':
        config = UNetConfig(**config)
        model = EnsembleUNet(config).to(device)
    elif model_name == 'deeplabv3plus': 
        model = DeepLabV3Plus().to(device)
    elif model_name == 'segnet':
        model = SegNet().to(device)
    elif model_name == 'fcn': 
        model = FCN_Vgg16_16s().to(device)
    elif model_name == "segformer": 
        print("initializing segformer")
        config = SegformerConfig(**config)
        model = SegFormer(config).to(device)
        print("segformer initialized")
    return model