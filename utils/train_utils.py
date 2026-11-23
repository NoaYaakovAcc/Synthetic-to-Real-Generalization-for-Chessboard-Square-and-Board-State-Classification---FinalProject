"""
This file contains functions for training and validating ChessNet.

The training and validation process is:
1. Load batches of chessboard images and their labels.
2. Pass the images through the model to obtain class scores.
3. Calculate the classification loss using the selected criterion.
4. Update the model weights during training.
5. Calculate square accuracy, non-empty accuracy, and perfect-board accuracy.
6. Return the average loss and accuracy metrics for the epoch.

When an optimizer is provided, it updates the model weights.
During validation, the model is evaluated without gradient calculations or weight updates.

"""
import torch
import json
from pathlib import Path
from config import EMPTY_CLASS


def _batch_metrics(logits, labels, empty_class=EMPTY_CLASS):
    
    """
    Calculates classification metrics for a single batch.

    Parameters:
        logits: Model output scores with shape (B, 64, 13).
        labels: Ground-truth class indices with shape (B, 64).
        empty_class: Class index representing an empty square.

    Returns:
        dict: Counts of correct squares, non empty squares, perfect boards and their totals.

    """

    predictions = logits.argmax(dim=-1)

    correct = predictions == labels
    non_empty = labels != empty_class

    square_correct = correct.sum().item()
    square_total = labels.numel()

    non_empty_correct = (correct & non_empty).sum().item()
    non_empty_total = non_empty.sum().item()

    perfect_boards = correct.all(dim=1).sum().item()
    board_total = labels.shape[0]

    metrics = {
        "square_correct": square_correct,
        "square_total": square_total,
        "non_empty_correct": non_empty_correct,
        "non_empty_total": non_empty_total,
        "perfect_boards": perfect_boards,
        "board_total": board_total
    }

    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):

    """
    Runs one complete training or validation epoch.

    When an optimizer is provided, the function performs backpropagation and updates the model weights.
    if not it evaluates the model with no gradients.

    Parameters:
        model: ChessNet model to train or evaluate.
        loader: DataLoader containing images, labels, and paths.
        criterion: Loss function used to evaluate predictions.
        device: Device used for computation (cpu or cuda).
        optimizer: Optional optimizer for training and  none,runs in validation mode.

    Returns:
        dict: Average loss, square accuracy, non-empty accuracy, and perfect-board accuracy for the epoch.

    """
    
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_images = 0

    square_correct = 0
    square_total = 0
    non_empty_correct = 0
    non_empty_total = 0
    perfect_boards = 0
    board_total = 0

    with torch.set_grad_enabled(is_train):
        
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)

            predictions_for_loss = logits.reshape(-1, logits.shape[-1])
            labels_for_loss = labels.reshape(-1)
            loss = criterion(predictions_for_loss, labels_for_loss)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_images += batch_size

            metrics = _batch_metrics(logits, labels)
            square_correct += metrics["square_correct"]
            square_total += metrics["square_total"]
            non_empty_correct += metrics["non_empty_correct"]
            non_empty_total += metrics["non_empty_total"]
            perfect_boards += metrics["perfect_boards"]
            board_total += metrics["board_total"]

    results = {
        "loss": total_loss / max(total_images, 1),
        "square_accuracy": square_correct / max(square_total, 1),
        "non_empty_accuracy": non_empty_correct / max(non_empty_total, 1),
        "perfect_board_accuracy": perfect_boards / max(board_total, 1)
    }

    return results


"""
Wrapper functions for run_epoch, to separate interfaces

"""

def train_one_epoch(model, loader, optimizer, criterion, device):

    """
    Trains the model for one epoch.

    Parameters:
        model: ChessNet model to train.
        loader: DataLoader containing training samples.
        optimizer: Optimizer used to update model weights.
        criterion: Loss function used during training.
        device: Device used for computation (CPU or CUDA).

    Returns:
        dict: Training loss and classification accuracy metrics.

    """
    return run_epoch(model, loader, criterion, device, optimizer)


def validate(model, loader, criterion, device):

    """
    Evaluates the model one validation epoch.
    (model weights are not being updated during validation.)

    Parameters:
        model: ChessNet model to evaluate.
        loader: DataLoader containing validation samples.
        criterion: Loss function used for validation.
        device: Device used for computation (cpu or cuda).

    Returns:
        dict: Validation loss and classification accuracy metrics.

    """
    return run_epoch(model, loader, criterion, device)
