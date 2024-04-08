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

## Reproduce our results 
### Use our pretrained weights 
Run the following code block to run evaluation on our test set with all our trained models used in the report. 
```
cd evaluation 
python run_evaluation.py 
```

### Retrain models on your own 
#### Autoencoder
```
python train.py --model_name=simpleunet
```
#### Unet 
```
python train.py --model_name=unet
```
#### Pix2Pix 
```
python train.py --model_name=unet
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