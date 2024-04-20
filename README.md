# 60.001 ADL Project 

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

## Replicate our results 
Download weights from our [google drive checkpoints](https://drive.google.com/file/d/1yhBQPDkkxOVsdExH7EyhQeNTPtA7gfU2/view?usp=share_link) and replace checkpoints directory

Compute the results from our project by running the following scripts:
```
cd evaluation 
python run_evaluation.py 
```

## Train model (python script)
```
python train.py --model_name=unet
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

## Evaluation
Paste relative directory of the checkpoint paths in evaluation/final_models.json, an example can be found in evaluation/final_models_example.json
```
cd evaluation 
python run_evaluation.py 
```