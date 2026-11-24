"""
This file evaluates ChessNet predictions and saves evaluation results.

The evaluation process is:
1. Runs the trained model on labeled chessboard images.
2. Compares the predicted classes with the ground-truth labels.
3. Calculates square accuracy, non-empty accuracya and perfect-board accuracy.
4. Builds a confusion matrix (heatmap) for the 13 chessboard classes.
5. Calculates precision, recall, F1 and macro F1 scores.
6. Save per-class metrics, confusion matrics and prediction visualizations.

The file also supports evaluating ilp postprocessing:
1. Generates board predictions using Argmax and ILP constraints.
2. Compares both methods against the ground-truth labels.
3. Count errors corrected and errors introduced by ILP.
4. Calculates the net number of corrected squares.
5. Records solver failures and use Argmax as a fallback if needed.
6. Saves ILP comparison results and visualizations.

The evaluation functions do not update the model weights.

"""

import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from config import (
    NUM_CLASSES, EMPTY_CLASS, BOARD_SIZE,
    CLASS_NAMES, FULL_CLASS_NAMES, IDX_TO_PIECE, FEN_TO_UNICODE,
    IMAGENET_MEAN, IMAGENET_STD, NUM_VISUALIZATIONS
)

from pre_post_processing.ilp_constraints import solve_board

def square_labels_to_board_string(square_labels):
    
    """
    Represents 64 class lables as eight rows of piece symbols.
    
    Parameters:
        square_labels: Tensor containing 64 class labels in row order from a8 to h1.
    
    Returns:
        str: String with eight rows, empty squares are dots.
    
    """

    # Disconnect from gradients
    square_labels = square_labels.detach() 

    square_labels = square_labels.cpu()       
    square_labels = square_labels.reshape(-1)  
    values = square_labels.tolist()   

    rows = []
       
    for row_number in range(BOARD_SIZE):
        row_text = ""
        for col_number in range(BOARD_SIZE):
            index = row_number * BOARD_SIZE + col_number
            piece_number = int(values[index])
            row_text += IDX_TO_PIECE.get(piece_number, "?")
        rows.append(row_text)

    return "/".join(rows)


def draw_board(ax, board_string, title):

    """
    Draws a chessboard and its pieces on an existing Matplotlib axis.
    
    Parameters:
        ax: Matplotlib axis on which to draw.
        board_string: Eight rows of piece letters and dots.
        title: Text shown above the board.
    
    Returns:
        none
    
    """
    rows = board_string.split("/")

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):

            
            if (row + col) % 2 == 0:
                color = "white"
            else:
                color = "gray"

            square = patches.Rectangle(
                (col, BOARD_SIZE - 1 - row), 
                1, 
                1,
                facecolor=color, 
                edgecolor="black"
            )
            ax.add_patch(square)

            
            piece = rows[row][col]
            symbol = FEN_TO_UNICODE.get(piece, "")

            # if not an empty square
            if symbol != "":
                ax.text(
                    col + 0.5, 
                    BOARD_SIZE - 0.5 - row, 
                    symbol,
                    ha="center", 
                    va="center", 
                    fontsize=25
                )

    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)


def save_prediction_visualization(image_tensor, predicted_labels, true_labels, save_path):

    """
    Saves an image comparing the input, predicted board, and true board.
    
    Parameters:
        image_tensor: Normalized input image tensor -> shape (3, H, W).
        predicted_labels : Predicted class labels for 64 squares.
        true_labels: Correct class labels for the same 64 squares.
        save_path: File path for the output png image.
    
    Returns:
        none
        
    """

    image = image_tensor.detach()
    image = image.cpu()

    #denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image = image * std + mean

    # reorganize the order of the Channels, Height, Width 
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    predicted_board = square_labels_to_board_string(predicted_labels)
    true_board = square_labels_to_board_string(true_labels)

    predictions = predicted_labels.detach()
    predictions = predictions.cpu()
    predictions = predictions.reshape(-1)

    labels = true_labels.detach()
    labels = labels.cpu()
    labels = labels.reshape(-1)

    accuracy = (predictions == labels).float()
    accuracy = accuracy.mean()
    accuracy = accuracy.item()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    draw_board(axes[1], predicted_board, f"Prediction\nSquare accuracy: {accuracy:.1%}")
    draw_board(axes[2], true_board, "Ground Truth")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_image(confusion, output_path):
    
    """
    Saves a heatmap of the row-normalized confusion matrix.
    
    Parameters:
        confusion: Tensor with true classes as rows and predicted classes as columns.
        output_path: File path for the heatmap confusion matrix image.
    
    Returns:
        none
    
    """

    matrix = confusion.detach()
    matrix = matrix.cpu()
    matrix = matrix.numpy().astype(float)

    normalized = np.zeros_like(matrix)

    for row in range(NUM_CLASSES):
        row_total = matrix[row].sum()
        if row_total > 0:
            normalized[row] = matrix[row] / row_total

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalized)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix")
    fig.colorbar(image, ax=ax)

    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            value = normalized[row, col]
            if value >= 0.01:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def calculate_f1_scores(confusion):

    """
    Calculates precision, recall, F1 and macro-F1 from the confusion matrix.
    
    Parameters:
        confusion: Confusion matrix containing counts for the 13 classes.
    
    Returns:
        tuple[float, list[float], list[float], list[float]]: Tuple (macro_f1, precision_scores, recall_scores, f1_scores).
    
    """

    matrix = confusion.float()
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for class_number in range(NUM_CLASSES):
        tp = matrix[class_number, class_number].item()
        fp = matrix[:, class_number].sum().item() - tp
        fn = matrix[class_number, :].sum().item() - tp

        precision = 0.0
        if tp + fp > 0:
            precision = tp / (tp + fp)

        recall = 0.0
        if tp + fn > 0:
            recall = tp / (tp + fn)

        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)
    return macro_f1, precision_scores, recall_scores, f1_scores


@torch.no_grad()
def evaluate_model(model, loader, device, output_dir=None,
                   num_visualizations=NUM_VISUALIZATIONS, curves_dir=None, prediction_dir=None):
    """
    Evaluates ChessNet over all batches and optionally saves results.

    Parameters:
        model: Trained chessboard classification model.
        loader: DataLoader yielding images, 64 labels, and image paths.
        device: CPU or CUDA device used for inference.
        output_dir: Optional directory for per-class metrics and evaluation data.
        num_visualizations: Maximum number of prediction examples to save.
        curves_dir: Optional folder for the confusion matrix image.
        prediction_dir: Optional folder for the prediction example images.

    Returns:
        tuple[dict, torch.Tensor]: Final metrics and the confusion matrix.

    Uses Argmax predictions. When output_dir is provided, also saves the
    per-class CSV, confusion matrix image, and selected prediction examples.
    """
    model.eval()

    # Rows are the true classes, columns are the predicted classes.
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    total_boards = 0
    perfect_boards = 0
    total_squares = 0
    correct_squares = 0
    non_empty_total = 0
    non_empty_correct = 0
    visualizations_saved = 0

    # creates output folders only if needed 
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        visualization_dir = Path(prediction_dir) if prediction_dir else output_dir / "predictions"
        visualization_dir.mkdir(parents=True, exist_ok=True)

    for images, labels, paths in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=-1)
        correct = predictions == labels

        total_squares += labels.numel()
        correct_squares += correct.sum().item()

        non_empty = labels != EMPTY_CLASS
        non_empty_total += non_empty.sum().item()
        non_empty_correct += (correct & non_empty).sum().item()

        total_boards += labels.shape[0]
        perfect_boards += correct.all(dim=1).sum().item()

        true_values = labels.reshape(-1).cpu()
        predicted_values = predictions.reshape(-1).cpu()

        for index in range(len(true_values)):
            true_class = int(true_values[index])
            predicted_class = int(predicted_values[index])
            confusion[true_class, predicted_class] += 1

        if output_dir is not None:
            for index in range(images.shape[0]):
                if visualizations_saved >= num_visualizations:
                    break

                filename = f"prediction_{visualizations_saved:02d}.png"
                save_path = visualization_dir / filename
                save_prediction_visualization( images[index], predictions[index], labels[index], save_path)
                visualizations_saved += 1

    square_accuracy = correct_squares / max(total_squares, 1)
    non_empty_accuracy = non_empty_correct / max(non_empty_total, 1)
    perfect_board_accuracy = perfect_boards / max(total_boards, 1)

    macro_f1, precision_scores, recall_scores, f1_scores = calculate_f1_scores(confusion)

    metrics = {
        "square_accuracy": square_accuracy,
        "non_empty_accuracy": non_empty_accuracy,
        "perfect_board_accuracy": perfect_board_accuracy,
        "macro_f1": macro_f1,
        "num_boards": total_boards
    }

    if output_dir is not None:
        confusion_path = Path(curves_dir) if curves_dir else output_dir
        confusion_path.mkdir(parents=True, exist_ok=True)
        save_confusion_matrix_image(confusion, confusion_path / "confusion_matrix.png")

        with open(output_dir / "per_class_metrics.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["class", "precision", "recall", "f1"])

            for index in range(NUM_CLASSES):
                writer.writerow([
                    FULL_CLASS_NAMES[index],
                    precision_scores[index],
                    recall_scores[index],
                    f1_scores[index]
                ])

    print("\nEvaluation summary:")
    print(f"Square accuracy: {square_accuracy:.4f}")
    print(f"Non-empty accuracy: {non_empty_accuracy:.4f}")
    print(f"Perfect-board accuracy: {perfect_board_accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    return metrics, confusion

def save_ilp_visualization(image_tensor, true_labels, argmax, ilp, save_path):

    """
    Saves a visual comparison of Argmax and ILP predictions.

    Parameters:
        image_tensor: Normalized input board image tensor.
        true_labels: Ground-truth class labels for all 64 squares.
        argmax: Class labels predicted using Argmax.
        ilp: Class labels predicted using the ILP constraints.
        save_path: Output path for the PNG image.

    Returns:
        None

    """
    image = image_tensor.detach()
    image = image.cpu()

    #denormalize
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image = image * std + mean

    # reorganize the order of the Channels, Height, Width 
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    argmax_accuracy = (argmax == true_labels).float()
    argmax_accuracy = argmax_accuracy.mean()
    argmax_accuracy = argmax_accuracy.item()

    ilp_accuracy = (ilp == true_labels).float()
    ilp_accuracy = ilp_accuracy.mean()
    ilp_accuracy = ilp_accuracy.item()

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    draw_board(axes[1], square_labels_to_board_string(true_labels), "Ground Truth")
    draw_board(axes[2], square_labels_to_board_string(argmax), f"Argmax\nSquare accuracy: {argmax_accuracy:.1%}")
    draw_board(axes[3], square_labels_to_board_string(ilp), f"ILP\nSquare accuracy: {ilp_accuracy:.1%}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_ilp(model, test_loader, device, output_dir,
                 prediction_dir=None, num_visualizations=5):
    """
    Compares Argmax with the full chess-constraint ILP solver.

    Parameters:
        model: Trained ChessNet with its best validation weights.
        test_loader: DataLoader containing labeled real chessboard images.
        device: CPU or CUDA device used for model inference.
        output_dir: Folder for the saved ILP comparison results.
        prediction_dir: Optional folder for ILP comparison images.
        num_visualizations: Maximum number of comparison images to save.

    Returns:
        dict: Accuracy of Argmax and ILP, fixed and introduced errors,
        and the number of solver failures.

    """
    model.eval()

    results_count = {
        "argmax": {"correct": 0, "occupied_correct": 0, "perfect": 0},
        "ilp_constraints": {"correct": 0, "occupied_correct": 0, "perfect": 0}
    }

    total_squares = 0
    total_occupied = 0
    total_boards = 0
    fixed_errors = 0
    introduced_errors = 0
    solver_failures = 0
    saved = 0

    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for images, labels, paths in test_loader:
        logits_batch = model(images.to(device)).cpu()

        for index, image_path in enumerate(paths):
            logits = logits_batch[index]
            actual = labels[index].cpu().reshape(-1)
            argmax = logits.argmax(dim=-1)
            
            try:
                ilp = torch.as_tensor(solve_board(logits), dtype=torch.int64)
            except RuntimeError as error:
                print("ILP failed for", image_path, ":", error)
                solver_failures += 1
                ilp = argmax.clone()

            occupied = actual != EMPTY_CLASS
            argmax_correct = argmax == actual
            ilp_correct = ilp == actual

            for method, correct in (("argmax", argmax_correct), ("ilp_constraints", ilp_correct)):
                results_count[method]["correct"] += int(correct.sum())
                results_count[method]["occupied_correct"] += int(correct[occupied].sum())
                results_count[method]["perfect"] += int(correct.all())

            fixed_errors += int((~argmax_correct & ilp_correct).sum())
            introduced_errors += int((argmax_correct & ~ilp_correct).sum())
            total_squares += actual.numel()
            total_occupied += int(occupied.sum())
            total_boards += 1

            if prediction_dir is not None and saved < num_visualizations:
                save_ilp_visualization(images[index], actual, argmax, ilp, prediction_dir / f"ilp_comparison_{saved:02d}.png")
                saved += 1

    results = {}
    for method, counts in results_count.items():
        results[method] = {
            "square_accuracy": counts["correct"] / max(total_squares, 1),
            "non_empty_accuracy": counts["occupied_correct"] / total_occupied if total_occupied else None,
            "perfect_board_accuracy": counts["perfect"] / max(total_boards, 1),
            "num_boards": total_boards
        }

    report = {
        "results": results,
        "fixed_errors": fixed_errors,
        "introduced_errors": introduced_errors,
        "net_corrected_squares": fixed_errors - introduced_errors,
        "solver_failures": solver_failures
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "ilp_results.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nILP comparison:")
    print(json.dumps(report, indent=2))
    
    return report
