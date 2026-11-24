"""
This file provides functions for predicting chessboard positions using a trained ChessNet model.

The prediction process is:
1. Loads a trained model checkpoint or reuse a previously loaded model.
2. Preprocesses an RGB chessboard image.
3. Passes the image through the model to get the class scores.
4. Selects the highest-scoring class for each of the 64 squares.
5. Return an 8x8 tensor containing the predicted class indices.

The file also provides functions for:
1. Converting predicted class indices into fen piece placement.
2. Creating an svg visualization of the predicted chessboard.

The loaded model is cached to avoid reloading the same checkpoint for every prediction.

"""

from pathlib import Path
import chess
import chess.svg
import numpy as np
import torch
import argparse
from PIL import Image

from config import MODEL_PATH, NUM_CLASSES, IDX_TO_PIECE
from model.model import ChessNet
from pre_post_processing.preprocessing import preprocess_numpy_image
from pre_post_processing.ilp_constraints import solve_board
from utils.run_utils import load_weights



MODEL = None
LOADED_CHECKPOINT = None


def load_model(checkpoint=None):
    """
    Loads a trained ChessNet model and re uses it for later predictions.

    Parameters:
        checkpoint: Path to the model checkpoint. Uses MODEL_PATH if omitted.

    Returns:
        ChessNet: Trained model in evaluation mode.

    """
    global MODEL, LOADED_CHECKPOINT

    model_path = checkpoint if checkpoint is not None else MODEL_PATH

    if model_path is None:
        raise ValueError( "No checkpoint provided. please set MODEL_PATH in config.py or use --checkpoint.")

    checkpoint_path = str(Path(checkpoint or MODEL_PATH).resolve())

    if MODEL is not None and LOADED_CHECKPOINT == checkpoint_path:
        return MODEL

    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    load_weights(model, checkpoint_path, device)
    model.eval()

    MODEL = model
    LOADED_CHECKPOINT = checkpoint_path
    return MODEL


def predict_board(image: np.ndarray, checkpoint=None, method="ilp"):

    """
    Predicts the contents of the 64 chessboard squares.

    Parameters:
        image: RGB chessboard image as a NumPy array.
        checkpoint: Optional path to a trained model checkpoint.
        method: Prediction method, "ilp" or "argmax" and uses ILP by default.
                

    Returns:
        torch.Tensor: cpu tensor of shape (8, 8) with integer class ids.

    """
    model = load_model(checkpoint)
    device = next(model.parameters()).device
    image_tensor = preprocess_numpy_image(image).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)[0]

    if method == "ilp":
        try:
            predictions = solve_board(logits)
            predictions = torch.as_tensor(predictions,dtype=torch.int64)

        except RuntimeError as error:
            print("ILP failed, fallback Argmax:", error)
            predictions = logits.argmax(dim=-1)

    elif method == "argmax":
        predictions = logits.argmax(dim=-1)

    else:
        raise ValueError("Unknown prediction method: " + method)

    return predictions.reshape(8, 8).cpu().long()


def prediction_to_fen(board_prediction: torch.Tensor):

    """
    Converts predicted class indices into FEN piece placement.

    Parameters:
        board_prediction: Tensor of shape (8, 8) with class indices.

    Returns:
        str: Piece-placement section of a fen string.

    """
    if board_prediction.shape != (8, 8):
        raise ValueError("Prediction must have shape (8, 8)")

    fen_rows = []
    for row in board_prediction.tolist():
        fen_row = ""
        empty_squares = 0

        for class_index in row:
            piece = IDX_TO_PIECE[int(class_index)]
            if piece == ".":
                empty_squares += 1
                continue

            if empty_squares:
                fen_row += str(empty_squares)
                empty_squares = 0

            fen_row += piece

        if empty_squares:
            fen_row += str(empty_squares)

        fen_rows.append(fen_row)

    return "/".join(fen_rows)


def save_board_image(prediction: torch.Tensor, output_path="predicted_board.svg"):
    
    """
    Saves an svg visualization of a predicted chessboard.

    Parameters:
        prediction: Tensor of shape (8, 8) with predicted class ids.
        output_path: Location where the SVG file will be saved.

    Returns:
        str: Path to the saved svg file.

    """

    piece_placement = prediction_to_fen(prediction)
    board = chess.Board(f"{piece_placement} w - - 0 1")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(chess.svg.board(board=board), encoding="utf-8")
    
    return str(output_file)


def main():
    """
    Runs chessboard prediction from the command line.
    """



    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument( "--output", default="predicted_board.svg")
    parser.add_argument("--method", choices=["ilp", "argmax"],default="ilp")

    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))

    prediction = predict_board(image,checkpoint=args.checkpoint,method=args.method)

    print("Predicted board:")
    print(prediction)

    print("\nFEN piece placement:")
    print(prediction_to_fen(prediction))

    saved_path = save_board_image(prediction, args.output)
    print("\nSaved visualization:", saved_path)


if __name__ == "__main__":
    main()
