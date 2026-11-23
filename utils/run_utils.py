"""
This file contains helper functions for managing training experiments and saving their results.

its main responsibilities are:
1. Set random seeds for Python, NumPy, and PyTorch.
2. Load model weights from saved checkpoints.
3. Read training history from JSON files.
4. Extract training and validation metrics across epochs.
5. Save loss and accuracy curves.
6. Save training, validation, and test image paths in splits.json.
7. Evaluate the final model and save its metrics.
8. Generate unique output directories for experiments.

These functions help organize experiments, track training
progress, and preserve information needed to repeat runs.
"""

import json
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.eval_utils import evaluate_model

def set_seed(seed):

    """
    Sets random seeds .
    
    Parameters:
        seed: Integer random seed for Python, NumPy, and PyTorch.
    
    Returns:
        none
    
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_weights(model, path, device):
    
    """
    Loads saved weights into the given model.
    
    Parameters:
        model: ChessNet instance to update.
        path: Path to a saved state dictionary or checkpoint.
        device: Device to map loaded tensors onto.
    
    Returns:
        None

    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        weights = checkpoint["model_state_dict"]
    else:
        weights = checkpoint

    model.load_state_dict(weights)


def read_history(path):

    """
    Reads saved per epoch training information.
    
    Parameters:
        path: Path to a history.json file.
    
    Returns:
        list[dict]: Non-empty list of dictionaries, one for each epoch.
        
    """

    with open(path, "r", encoding="utf-8") as file:
        history = json.load(file)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("History must be a non  empty list")

    return history


def metric(history, phase, key):

    """
    Extracts one metric across all epochs.
    
    Parameters:
        history: List of dictionaries describing epochs.
        phase: "train" or "val".
        key: Metric name, such as "loss" or "square_accuracy".
    
    Returns:
        list[float]: List of numeric metric values in epoch order.
    
    """

    values = []

    for entry in history:
        if phase == "val" and "validation" in entry:
            data = entry["validation"]
        else:
            data = entry[phase]
        values.append(float(data[key]))

    return values

def save_training_curves(history, output_dir, stage_name, val_description=None):

    """
    Saves the training and validation loss and accuracy plots.

    Parameters:
        history: Recorded results for each training epoch.
        output_dir: Experiment directory containing the curves folder.
        stage_name: Training mode used as the title of the plots.
        val_description: Optional description of the validation data.

    Returns:
        None. 

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [entry["epoch"] for entry in history]
    title = stage_name.replace("_", " ").title()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metric(history, "train", "loss"), label="Train loss")
    plt.plot(epochs, metric(history, "val", "loss"), label="Validation loss")
    plt.title(title + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, metric(history, "train", "square_accuracy"), label="Train per square")
    plt.plot(epochs, metric(history, "val", "square_accuracy"), label="Validation per square")
    plt.plot(epochs, metric(history, "train", "non_empty_accuracy"), label="Train per non-empty")
    plt.plot(epochs, metric(history, "val", "non_empty_accuracy"), label="Validation per non-empty")
    plt.title(title + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png")
    plt.close()


def save_splits(args, train_samples, val_samples, test_samples, output_dir):

    """
    Saves the lists of train, validation and test image paths in splits.json.

    Parameters:
        args: Parsed experiment settings
        train_samples: Training sample list.
        val_samples: Internal validation sample list.
        test_samples: Final real test sample list.
        output_dir: Directory where splits.json is saved.
    
    Returns:
        none

    """
    split_info = {
        "mode": args.mode,
        "split_method": "fen-position groups; separate test games",
        "val_fraction": args.val_fraction,
        "test_games": args.test_games,
        "train": [],
        "epoch_validation": [],
        "final_real_test": []
    }

    for sample in train_samples:
        split_info["train"].append(sample.img_path)
    for sample in val_samples:
        split_info["epoch_validation"].append(sample.img_path)
    for sample in test_samples:
        split_info["final_real_test"].append(sample.img_path)

    with open(Path(output_dir) / "training_info" / "splits.json", "w", encoding="utf-8") as file:
        json.dump(split_info, file, indent=2)


def evaluate_and_save(model, test_loader, criterion, device, output_dir, args, best_epoch):

    """
    Runs the final real-data evaluation and saves the results.

    Parameters:
        model: Trained ChessNet with its best validation weights.
        test_loader: DataLoader containing the real evaluation images.
        criterion: Training loss function (kept for compatibility).
        device: CPU or CUDA device used for evaluation.
        output_dir: Main experiment output directory.
        args: Parsed training arguments.
        best_epoch: Epoch selected by internal validation.

    Returns:
        dict: Final evaluation metrics, without test loss.

    """

    output_dir = Path(output_dir)
    metrics, _ = evaluate_model(
        model, 
        test_loader,
        device,
        output_dir=output_dir / "training_info",
        curves_dir=output_dir / "curves",
        prediction_dir=output_dir / "prediction_samples"
    )

    with open(output_dir / "training_info" / "final_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nFINAL REAL TEST:")

    for key, value in metrics.items():
        print(key, ":", value)

    return metrics


def experiment_output_dir(args):

    """
    Chooses a unique output folder for the experiment.

    Parameters:
        args: Parsed arguments: training mode, optimizer, epochs, training games, optional freezing choice and an output path.

    Returns:
        pathlib.Path: A folder path that does not already exist.

    Uses  user-provided output directory when available. 
    Otherwise, builds a name from the experiment settings and adds a numeric suffix
    if a folder with the same name already exists.
    Its on the developers part to follow.

    """

    if args.output_dir:
        folder = Path(args.output_dir)
    else:
        mode = args.mode
        if mode == "finetune":
            freeze = "frozen" if args.freeze_backbone else "partial" if args.partial_freeze else "full"
            mode += "_" + freeze

        games = ""
        if args.mode in ("zero_shot", "combined"):
            games += "s" + "".join(map(str, sorted(args.synthetic_games)))
        if args.mode in ("finetune", "combined"):
            games += "r" + "".join(map(str, sorted(args.real_games)))
        folder = Path("experiments") / f"{mode}_{args.optimizer}_{args.epochs}_{games}"

    if not folder.exists():
        return folder

    number = 1
    while folder.with_name(folder.name + f"_{number}").exists():
        number += 1

    return folder.with_name(folder.name + f"_{number}")
