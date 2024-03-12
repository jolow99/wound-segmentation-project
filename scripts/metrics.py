
import torch 
from torchmetrics.functional import dice, accuracy, precision, recall
from torchmetrics import Dice
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall

def compute_metrics(eval_predictions):
    # Compute the accuracy
    dice_metric = Dice()    
    binary_accuracy = BinaryAccuracy()
    binary_precision = BinaryPrecision()
    binary_recall = BinaryRecall()

    for i, (pred, target) in enumerate(zip(eval_predictions.predictions, eval_predictions.label_ids)):
        pred = torch.tensor(pred)
        target = torch.tensor(target).to(torch.int)
        dice_metric.update(pred, target)
        binary_accuracy.update(pred, target)
        binary_precision.update(pred, target)
        binary_recall.update(pred, target)

    accuracy = binary_accuracy.compute().item()
    precision = binary_precision.compute().item()
    recall = binary_recall.compute().item()
    dice = dice_metric.compute().item()

    results = {
        "dice": dice,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    print(results)
    return results

if __name__=="__main__":
    torch.manual_seed(42)
    pred = torch.rand((3,4,4))
    target = torch.randint(0, 2, (3,4,4))
    print("Pred: ", pred)
    print("Target: ", target)
    metrics = compute_metrics(pred, target)
    print("Metrics: ",  metrics)
