
import sys 
import torch
sys.path.append('..')
from models.unet.modeling_simple_unet import SimpleUNet
from models.unet.modeling_unet import UNet
from .training_args import TrainingArguments
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from utils.utils import EvalPrediction
import numpy as np

class Trainer: 
    def __init__(
        self, 
        model, 
        train_loader, 
        validation_loader=None, 
        test_loader=None,
        compute_metrics=None,
        device="cuda", 
        args=TrainingArguments(),
        criterion=torch.nn.BCELoss()
    ): 
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.model = model
        self.compute_metrics = compute_metrics
        self.criterion = criterion
        self.args = args
        self.device = device
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f'{self.args.logdir}/{self.args.expt_name}/{self.timestamp}')
        self.checkpoint_path = f'{self.args.checkpoint_path}/{self.args.expt_name}/{self.timestamp}'

    def train(self): 
        optimizer = self.args.optim(self.model.parameters(), lr=self.args.learning_rate)
        # criterion = torch.nn.BCELoss()
        criterion = self.criterion
        
        best_validation_loss = float('inf')
        
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        print("logging path: ", f'{self.args.logdir}/{self.args.expt_name}/{self.timestamp}')
        print("checkpoint path: ", self.checkpoint_path)
        
        with tqdm(range(self.args.num_train_epochs), colour="green", desc="Epochs", leave=True) as pbar:
            for epoch in range(self.args.num_train_epochs):
                pbar.set_description(f"Epoch {epoch+1}/{self.args.num_train_epochs}")
                train_loss = 0
                with tqdm(range(len(self.train_loader)), colour="magenta", desc=f"Epoch {epoch}", leave=False) as epoch_pbar:
                    self.model.train()
                    for batch_number, (inputs, labels) in enumerate(self.train_loader):
                        epoch_pbar.set_description(f"Batch {batch_number+1}/{len(self.train_loader)}")
                        epoch_pbar.update(1)
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * inputs.size(0)
                        epoch_pbar.set_postfix(loss=loss.item())

                    validation_loss = 0.0
                    self.model.eval()
                    with torch.no_grad():
                        for i, (inputs, labels) in enumerate(self.validation_loader):
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)
                            validation_loss += loss.item() * inputs.size(0)
                    self.writer.add_scalars('Training vs. Validation Loss',{'Training':train_loss,'Validation':validation_loss},epoch + 1)
                    self.writer.flush()
                    if validation_loss < best_validation_loss:
                        torch.save(self.model.state_dict(), f'{self.checkpoint_path}/best_model.pth')
                        best_validation_loss = validation_loss

                    pbar.update(1)
                print(f'\nEpoch {epoch+1}/{self.args.num_train_epochs} Train loss: {train_loss:.4f} Validation loss: {validation_loss:.4f}')
                # print("-----------------------------------------------------------------------------------")
            
            # dice_score = dice_coeff(outputs, labels)
            # train_dice += dice_score.item() * inputs.size(0)

            # f'Train Dice: {train_dice:.4f}')
    def evaluate(self): 
        self.model.eval()
        criterion = self.criterion
        outputs_all = []
        labels_all = []

        if self.compute_metrics is None: 
            test_loss = 0
            with tqdm(range(len(self.test_loader)), colour="magenta", desc="Testing", leave=True) as pbar:
                for i, (inputs, labels) in enumerate(self.test_loader):
                    pbar.update(1)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * inputs.size(0)
            overall_metrics = {"loss": test_loss}
        else: 
            with tqdm(range(len(self.test_loader)), colour="magenta", desc="Testing", leave=True) as pbar:
                for i, (inputs, labels) in enumerate(self.test_loader):
                    pbar.update(1)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    outputs_all.append(outputs.detach().cpu().numpy())
                    labels_all.append(labels.detach().cpu().numpy())

            overall_metrics = self.compute_metrics(EvalPrediction(np.concatenate(outputs_all), np.concatenate(labels_all), None))

        if self.writer is not None: 
            for k, v in overall_metrics.items(): 
                self.writer.add_scalar(k, v, 0)
            self.writer.flush()
        return overall_metrics
        



        

