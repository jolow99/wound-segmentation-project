from tabulate import tabulate
import argparse
import json 
import torch 
import os
import sys
sys.path.append('..')
from models.models import get_model
from scripts.dataloader import DataGen, create_dataloaders
from tqdm import tqdm
from scripts.metrics import compute_metrics
from utils.utils import EvalPrediction
import numpy as np
import matplotlib.pyplot as plt

    
# Read logged information
model_name = 'MyModel'
description = 'This is my model'
accuracy = 0.85  # Sample accuracy value
precision = 0.78  # Sample precision value
recall = 0.82  # Sample recall value
dice_score = 0.75  # Sample dice score value


# Define colors
green = '\033[92m'
red = '\033[91m'
end_color = '\033[0m'

# Create table
table = [
    ["Autoencoder", "0.78", "0.82", "0.85", "0.75"],
    ["UNET", "0.78", "0.84", "0.85", "0.75"],
    ["Pix2Pix", "0.78", "0.82", "0.85", "0.75"],
    ["Segformer", "0.92", "0.82", "0.65", "0.75"],
]

def print_table(data): 
    print(tabulate(data, headers=["Model Name", "Precision", "Recall", "Accuracy", "Dice"], tablefmt="fancy_grid"))

argparser = argparse.ArgumentParser(description="ADL Deep Learning Project")
argparser.add_argument("--data", type=str, default="../data/azh_wound_care_center_dataset_patches", help="Path to data")
argparser.add_argument("--device", type=str, default="mps", help="Device to use for training")
argparser.add_argument("--logdir", type=str, default="../logs", help="Path to save results")
argparser.add_argument("--checkpoint_path", type=str, default="../checkpoints", help="Path to save model")
argparser.add_argument("--batch_size", type=int, default=4, help="Batch size")

def evaluate_model_name(model_name, model_path, args): 
    print(f"Evaluating model: {model_name}")
    data_filepath = '../data/azh_wound_care_center_dataset_patches/'
    data_gen = DataGen(os.path.join(os.getcwd(), data_filepath), split_ratio=0.2)
    train_loader, validation_loader, test_loader = create_dataloaders(data_gen, args.batch_size, args.device)

    model = get_model(model_name, vars(args), device=args.device)
    model_checkpoint_path = f'{args.checkpoint_path}/{model_path}/best_model.pth'
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    criterion=torch.nn.BCELoss()
    outputs_all = []
    labels_all = []

    def save_image(input, output, label, filename):
        # concatenate input, output, label and save as image    
        input = input.squeeze(0)
        print(input.shape)
        input = input.permute(1,2,0).detach().cpu().numpy()
        output = output.squeeze(0)
        output = torch.stack([output, output, output], dim=0)
        output = torch.nn.Sigmoid()(output)
        # output = (output > 0.5).float()
        output = output.permute(1,2,0).detach().cpu().numpy()
        print(output)
        plt.imshow(output)
        print(output.shape)
        label = label.squeeze(0)
        label = torch.stack([label, label, label], dim=0)
        label = label.permute(1,2,0).detach().cpu().numpy()
        print(label.shape)
        output = np.concatenate([input, output, label], axis=1)
        if not os.path.exists("/".join(filename.split('/')[:-1])):
            os.makedirs("/".join(filename.split('/')[:-1]))
        plt.imsave(filename, output)

    with tqdm(range(len(test_loader)), colour="magenta", desc="Testing", leave=True) as pbar:
        for i, (inputs, labels) in enumerate(test_loader):
            # save 5 outputs to folder
            pbar.update(1)
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            if i < 5:
                save_image(inputs[0], outputs[0], labels[0], f'outputs/{model_name}_{i}_output.png')
                # save_image(labels[0], f'outputs/{model_name}_{i}_label.png')
            outputs_all.append(outputs.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())

    overall_metrics = compute_metrics(EvalPrediction(np.concatenate(outputs_all), np.concatenate(labels_all), None))
    return overall_metrics

if __name__ == "__main__":
    args = argparser.parse_args()
    model_runs = json.load(open('final_models.json'))
    table_data = []
    print("------------------------------------------------------------\n")
    for model_run in model_runs: 
        print("Evaluating model: ", model_run)
        model_name = model_run.split('_')[0]
        metrics = evaluate_model_name(model_name, model_run, args)
        print(f'{model_name} metrics: {metrics}')
        expt_name = model_run.split('/')[0]
        table_data.append([expt_name, metrics['precision'], metrics['recall'], metrics['accuracy'], metrics['dice']])

        # model = get_model(model, model_run, vars(args))
        # model.load_state_dict(torch.load(model_run))
        print("\n------------------------------------------------------------\n")
    print_table(table_data)


