# 60.004 ADL Project 

## Authors of this repository 
1. Clarence Lee, 1005266
2. Lai Pin Nean, 1005642
3. Joseph Low, 1005013

## Summary of directories
- models/ code files for all the model architectures and configurations 
- scripts/ code files for trainer script and dataloader 
- utils/ utility code files 
- data/ folder for all data files 
- checkpoints/ folder to store all model checkpoints 
- logs/ folder to store tensorboard logs 
- evaluation/ folder to store evaluation scripts 

## Setup 
Create environment 
```
conda create adl_project 
conda activate adl_project
```
Install dependencies
```
pip install -r requirements.txt
```

## Train model (python script)
```
python train.py --model_name=unet
```

## Evaluate your models  
Run the following code block to run evaluation on your previously trained models
```
cd evaluation 
python run_evaluation.py 
```

### Retrain models on your own 
#### Autoencoder
```
python train.py --model simpleunet --num_train_epochs 50
```
#### Unet 
```
python train.py --model unet --num_train_epochs 50
```
#### CircleNet
```
python train.py --model circlenet --num_train_epochs 50
```

#### MixCircleNet
```
python train.py --model_name=mixcirclenet --num_train_epochs 50
```
#### DeepLabV3Plus 
```
python train.py --model deeplabv3plus --num_train_epochs 50
```
#### SegNet
```
python train.py --model segnet --num_train_epochs 50
```
#### Fully Convolutional Network 
```
python train.py --model fcn --num_train_epochs 50 
```
#### Segformer 
```
python train.py --model segformer --num_train_epochs 10 
```


You can find your saved checkpoints and logs at 
- Logs: logs/{model_name}_{expt_name}/{timestamp}
- Checkpoints: checkpoints/{model_name}_{expt_name}/{timestamp}

#### Run evaluation
Paste all the checkpoint paths and logs in evaluation/final_models.json, an example can be found in evaluation/final_models_example.json
```
cd evaluation 
python run_evaluation.py 
```