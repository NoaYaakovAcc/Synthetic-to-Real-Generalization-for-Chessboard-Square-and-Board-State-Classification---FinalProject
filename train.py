"""
This file runs the main training and evaluation process for the chessboard square classification model.

The experiment process is:
1. Reads and validate the training settings from command-line arguments.
2. Creates an output directory and save the experiment settings.
3. Prepares the training, validation and test datasets with the option to reuse previously saved data splits.
4. Initializes the ChessNet and optionally load a trained checkpoint.
5. Configures the optimizer and the selected freezing strategy if needed.
6. Trains and validate the model for the requested number of epochs.
7. Saves training history, loss and accuracy curves  and checkpoints.
8. Selects and restore the best model based on validation non-empty accuracy, using validation loss to break ties if needed.
9. Evaluates the best model on the real test dataset.
10. Optionally compare Argmax predictions with ILP predictions if selected.

The file supports three training modes: Zero-shot, Fine-tuning, and Combined training.

Fine-Tuning also supports Full freeze, Partial freeze, and No freeze methods.

"""

import argparse
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn

from config import (
    NUM_CLASSES, EMPTY_CLASS, BATCH_SIZE, NUM_WORKERS, SEED, EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, VAL_FRACTION, DATA_ROOT,
    SYNTHETIC_GAMES, REAL_GAMES, TEST_GAMES, PIECE_LOSS_WEIGHT,
    EMPTY_LOSS_WEIGHT, SPLIT_ATTEMPTS,
)

from utils.data_utils import build_loader, prepare_samples,prepare_samples_from_saved_split
from utils.run_utils import set_seed, load_weights, save_training_curves, save_splits, evaluate_and_save, experiment_output_dir
from utils.train_utils import train_one_epoch, validate
from utils.eval_utils import evaluate_ilp
from model.model import ChessNet


def train_stage(model, train_loader, val_loader, device, args, val_description):

    """
    Trains one model and restores its best internal-validation checkpoint.
    
    Parameters:
        model: Model to train.
        train_loader: DataLoader with training boards.
        val_loader: DataLoader with internal-validation boards.
        device: cpu or cuda device.
        args: Parsed command-line arguments for training settings.
        val_description: Name of the internal validation set for plots.
    
    Returns:
        tuple[ChessNet, list[dict], torch.nn.Module, int]: Tuple (model, history, criterion, best_epoch); model contains the best weights.
        
    """

    output_dir = Path(args.output_dir)
    info_dir = output_dir / "training_info"
    models_dir = output_dir / "models"

    #weights = torch.tensor([PIECE_LOSS_WEIGHT] * (NUM_CLASSES - 1) + [EMPTY_LOSS_WEIGHT], dtype=torch.float32, device=device)
    #criterion = nn.CrossEntropyLoss(weight=weights)

    criterion = nn.CrossEntropyLoss()


    # Start with all layers trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    if args.freeze_backbone:

        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

        last_layer = model.backbone.classifier[2] 

        for parameter in last_layer.parameters():
            parameter.requires_grad = True

    elif args.partial_freeze:

        layers_to_freeze = [model.backbone.features[:6]] 

        for layer in layers_to_freeze:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    trainable_parameters = []

    for parameter in model.parameters():
        if parameter.requires_grad:
            trainable_parameters.append(parameter)

   
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )

    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY
        )

    best_accuracy = -1
    best_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = validate(model, val_loader, criterion, device)

        print("\nEpoch", epoch, "of", args.epochs)
        print("Train loss:", round(train_result["loss"], 4))
        print("Validation loss:", round(val_result["loss"], 4))
        print("Validation square accuracy:", round(val_result["square_accuracy"], 4))
        print("Validation non-empty accuracy:", round(val_result["non_empty_accuracy"], 4))
        print("Validation perfect-board accuracy:", round(val_result["perfect_board_accuracy"], 4))

        history.append({
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_result,
            "validation": val_result
        })

        with open(info_dir / "history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        save_training_curves(history, output_dir / "curves", args.mode, val_description)

        accuracy = val_result["non_empty_accuracy"]
        loss = val_result["loss"]


        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy = accuracy
            best_loss = loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, models_dir / "best_model.pth")
            print("Saved new best model")

    torch.save(model.state_dict(), models_dir / "last_model.pth")
    model.load_state_dict(best_weights)
    print("Best epoch:", best_epoch)
    return model, history, criterion, best_epoch

def check_arguments(parser, args):

    """
    Checks experiment settings before starting a training run.

    Parameters:
        parser: ArgumentParser
        args: Parsed experiment settings.

    Returns:
        None
    """
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error(
            "Epochs and batch size must be positive, ""workers cannot be negative")

    if args.val_fraction <= 0 or args.val_fraction >= 1:
        parser.error("Validation fraction must be between 0 and 1")

    if args.mode == "finetune" and args.checkpoint is None:
        parser.error("Fine-tuning needs --checkpoint")

    if args.freeze_backbone and args.partial_freeze:
        parser.error("Choose only one freezing option")

    if args.mode != "finetune" and (args.freeze_backbone or args.partial_freeze):
        parser.error("Freezing options are only for fine-tuning")

    if args.mode == "combined":
        for game in args.synthetic_games:
            if game in args.real_games:
                parser.error("Synthetic and real training games must be different")

    used_games = []
    if args.mode in ["zero_shot", "combined"]:
        used_games.extend(args.synthetic_games)
    if args.mode in ["finetune", "combined"]:
        used_games.extend(args.real_games)

    for game in args.test_games:
        if game in used_games:
            parser.error("Test games cannot also be training games")

def main():
    """
    Runs the main training and evaluation pipeline.

    Supports Zero-shot, Fine-tuning, and combined Training.

    Returns:
        none.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["zero_shot", "finetune", "combined"], required=True)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--synthetic-games", nargs="+", type=int, default=SYNTHETIC_GAMES)
    parser.add_argument("--real-games", nargs="+", type=int, default=REAL_GAMES)
    parser.add_argument("--test-games", "--val-games", dest="test_games", nargs="+", type=int, default=TEST_GAMES)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--splits-file", default=None)
   
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)

    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--partial-freeze", action="store_true")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--evaluate-ilp", action="store_true")

    args = parser.parse_args()

    check_arguments(parser, args)

    output_dir = experiment_output_dir(args)
    if args.splits_file:
        source = Path(args.splits_file).resolve()
        destination = (output_dir / "training_info" / "splits.json").resolve()
        if source == destination:
            parser.error("--output-dir must differ from the original split's experiment directory")

    args.output_dir = str(output_dir)
    for folder in ("models", "curves", "training_info", "prediction_samples"):
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    settings = vars(args).copy()
    settings["freeze_mode"] = ( "frozen" if args.freeze_backbone else "partial" if args.partial_freeze else "full") if args.mode == "finetune" else "none"
    settings["weight_decay"] = WEIGHT_DECAY
    settings["piece_loss_weight"] = PIECE_LOSS_WEIGHT
    settings["empty_loss_weight"] = EMPTY_LOSS_WEIGHT
    settings["backbone"] = "convnext_tiny"
    settings["loss_function"] = "CrossEntropyLoss "
    settings["augmentation"] = ( "synthetic_train_only: ""ColorJitter(p=0.8), GaussianBlur(p=0.2)")
    settings["split_attempts"] = SPLIT_ATTEMPTS

    with open(output_dir / "training_info" / "train_settings.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2)
    print("Output directory:", output_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.splits_file:
        train_samples, val_samples, test_samples, val_description = prepare_samples_from_saved_split(args)
    else:
        train_samples, val_samples, test_samples, val_description = prepare_samples(args)
    save_splits(args, train_samples, val_samples, test_samples, output_dir)

    train_loader = build_loader(
        train_samples,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    val_loader = build_loader(
        val_samples,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = build_loader(
        test_samples,
        train=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model = ChessNet(num_classes=NUM_CLASSES,pretrained=args.checkpoint is None).to(device)

    if args.checkpoint is not None:
        print("Loading checkpoint:", args.checkpoint)
        load_weights(model, args.checkpoint, device)

    model, history, criterion, best_epoch = train_stage(model, train_loader, val_loader, device, args, val_description)

    evaluate_and_save(model, test_loader, criterion, device, output_dir, args, best_epoch)

    if args.evaluate_ilp:
        evaluate_ilp(model, test_loader, device, output_dir / "training_info",
                     prediction_dir=output_dir / "prediction_samples", num_visualizations=5)

if __name__ == "__main__":
    main()
