import os 
import argparse 
from models.models import get_model
from scripts.trainer import Trainer
from scripts.training_args import TrainingArguments
from scripts.dataloader import DataGen, create_dataloaders
from scripts.metrics import compute_metrics

argparser = argparse.ArgumentParser(description="ADL Deep Learning Project")
argparser.add_argument("--data", type=str, default="./data/azh_wound_care_center_dataset_patches", help="Path to data")
argparser.add_argument("--model", type=str, default="unet", help="Model to use")
argparser.add_argument("--num_train_epochs", type=int, default=10, help="Number of epochs")
argparser.add_argument("--batch_size", type=int, default=8, help="Batch size")
argparser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
argparser.add_argument("--logdir", type=str, default="logs", help="Path to save results")
argparser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save model")
argparser.add_argument("--save_model", type=bool, default=True, help="Boolean value, set to true to save model during training")
# argparser.add_argument("--expt_name", type=str, default=None, help="Name of the saved model")
argparser.add_argument("--device", type=str, default="mps", help="Device to use for training")
argparser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
argparser.add_argument("--from_checkpoint", type=str, default=None, help="Path to checkpoint to load model from")
argparser.add_argument("--full_train_data", type=str, default=False, help="True if use full training data")

# python train.py --model unet

if __name__=="__main__": 
    expt_name = input("Enter name of expt [Press 'enter' to use default name temp]: ")
    if expt_name == "":
        expt_name = "temp"
    expt_description = input("Give a description of the experiment [Press 'enter' to skip]: ")
    args = argparser.parse_args()
    args.expt_name = args.model + "_" + expt_name
    args.expt_description = expt_description
    model = get_model(args.model, vars(args), device=args.device)
    data_filepath = './data/azh_wound_care_center_dataset_patches/'
    if args.full_train_data:
        data_gen = DataGen(os.path.join(os.getcwd(), data_filepath), split_ratio=0)
    else:
        data_gen = DataGen(os.path.join(os.getcwd(), data_filepath), split_ratio=0.2)
    train_loader, validation_loader, test_loader = create_dataloaders(data_gen, args.batch_size, args.device)
    trainer = Trainer(
        model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        args=TrainingArguments(**vars(args)),
        compute_metrics=compute_metrics,
        device=args.device
    )
    trainer.train()
    trainer.evaluate()
