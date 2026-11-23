"""
This file handles loading and preparing labeled chessboard data.

The data processing flow is:
1. Read the csv files and match frame numbers to fen labels.
2. Find chessboard images and identify whether they are synthetic or real.
3. Create a ChessBoardSample for each image with a matching fen.
4. Convert each FEN into 64 numeric class labels.
5. Load and transform images when requested by the Dataset.

The prepared images and labels are used for training and evaluation.

"""

import os
import re
import glob
from dataclasses import dataclass
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


from config import BOARD_SIZE, EMPTY_CLASS, PIECE_TO_INDEX


@dataclass
class ChessBoardSample:

    """
    Store one labeled chessboard image.
    
    Attributes:
        img_path: Path to the image.
        fen: Chess position in fen notation.
        domain: "synthetic" or "real".
        
    """
    img_path: str
    fen: str
    domain: str


def infer_domain(path):

    """
    Identify whether an image came from synthetic or real data.
    
    Parameters:
        path: File path of the image. Synthetic paths contain "generated"; real paths contain "tagged".
    
    Returns:
        str: "synthetic" or "real".
    
    """
    path = path.lower()

    if "generated" in path:
        return "synthetic"

    if "tagged" in path:
        return "real"

    raise ValueError("Unknown image type: " + path)


def extract_frame_number(path):
    
    """
    Reads the frame number from a filename such.

    Parameters:
        path: Image file path.
    
    Returns:
        int: Frame number as an integer, if is valid.
    
    """

    filename = os.path.basename(path).lower()
    match = re.search(r"frame[_-]?(\d+)", filename)

    if match is None:
        return None

    frame_number = int(match.group(1))
    return frame_number


def fen_to_labels(fen):

    """
    Converts a chess position in fen into 64 numeric class labels.
    
    Parameters:
        fen: fen string describing the chessboard.
    
    Returns:
        torch.Tensor: PyTorch tensor with 64 entries, ordered from a8 to h1.
    
    """

    board = str(fen).strip().split()[0]
    rows = board.split("/")

    if len(rows) != BOARD_SIZE:
        raise ValueError("fen must contain 8 rows: " + str(fen))

    labels = []

    for row in rows:
        row_labels = []

        for symbol in row:
            if symbol in "12345678":
                number_of_empty_squares = int(symbol)

                for i in range(number_of_empty_squares):
                    row_labels.append(EMPTY_CLASS)

            elif symbol in PIECE_TO_INDEX:
                row_labels.append(PIECE_TO_INDEX[symbol])

            else:
                raise ValueError("Unknown FEN symbol: " + symbol)

        if len(row_labels) != BOARD_SIZE:
            raise ValueError("Each fen row must have 8 squares: " + row)

        for label in row_labels:
            labels.append(label)

    return torch.tensor(labels, dtype=torch.long)


def load_frame_fens(csv_path):

    """
    Reads the csv and matchs frame numbers to fen strings.
    
    Parameters:
        csv_path: Path to a csv file containing from_frame, to_frame, and fen columns.
    
    Returns:
        dict[int, str]: Dictionary mapping each labeled frame number to its fen string.
    
    """

    table = pd.read_csv(csv_path)
    table.columns = table.columns.str.strip()

    required_columns = ["from_frame", "to_frame", "fen"]

    for column in required_columns:
        if column not in table.columns:
            raise ValueError("Missing CSV column: " + column)

    frame_fens = {}

    for index, row in table.iterrows():
        start = int(row["from_frame"])
        end = int(row["to_frame"])
        fen = str(row["fen"]).strip()

        if start != end:
            raise ValueError("Expected one frame per row: " + str(start) + " - " + str(end))

        if start in frame_fens:
            raise ValueError("Duplicate frame: " + str(start))

        # Check that the FEN is valid before using it.
        fen_to_labels(fen)
        frame_fens[start] = fen

    return frame_fens


def scan_game(game_root, csv_path):
    
    """
    Find image files in one game folder and attach their matching fen labels.
    
    Parameters:
        game_root: Root directory of the game images.
        csv_path: csv file containing the frame labels.
    
    Returns:
        list[ChessBoardSample]: List of ChessBoardSample objects for images with their matching csv.
    
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError("CSV not found: " + csv_path)

    frame_fens = load_frame_fens(csv_path)

    image_paths = []
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    for extension in extensions:
        pattern = os.path.join(game_root, "**", extension)
        files = glob.glob(pattern, recursive=True)

        for path in files:
            image_paths.append(path)

    samples = []
    skipped_no_frame = 0
    skipped_no_label = 0

    for path in sorted(set(image_paths)):
        frame = extract_frame_number(path)

        #personal check that all the images are processed
        if frame is None:
            skipped_no_frame += 1
            continue

        if frame not in frame_fens:
            skipped_no_label += 1
            continue

        sample = ChessBoardSample(
            img_path=path,
            fen=frame_fens[frame],
            domain=infer_domain(path)
        )
        samples.append(sample)

    print("Game folder:", game_root)
    print("Labeled images:", len(samples))
    print("Files without frame number:", skipped_no_frame)
    print("Frames without FEN:", skipped_no_label)

    return samples


class ChessBoardDataset(Dataset):
    
    """
    Provides labeled chessboard images to PyTorch during training and evaluation.
    
    Each item contains a processed image, its 64 fen labels, and its file path.
    
    """
    def __init__(self, samples, transform=None):

        """
        Creates a dataset from labeled samples.
        
        Parameters:
            samples: List of ChessBoardSample objects.
            transform: Optional to apply to each image.
        
        Returns:
            None
        
        """

        self.samples = list(samples)
        self.transform = transform

    def __len__(self):

        """
        Counts the labeled examples in this dataset.
        
        Parameters:
            None
        
        Returns:
            int: Number of samples as an integer.
        
        """
        return len(self.samples)

    def __getitem__(self, index):
        
        """
        Loads one chessboard image and its 64 class labels.
        
        Parameters:
            index: Index of the requested sample.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: Tuple of processed image, labels tensor, and original image path.
        
        """

        sample = self.samples[index]

        image = Image.open(sample.img_path).convert("RGB")
        labels = fen_to_labels(sample.fen)

        if self.transform is not None:
            image = self.transform( image, domain=sample.domain)

        return image, labels, sample.img_path


def split_by_domain(samples):

    """
    Separates labeled examples into synthetic and real groups.
    
    Parameters:
        samples: List of labeled ChessBoardSample objects.
    
    Returns:
        tuple[list[ChessBoardSample], list[ChessBoardSample]]: Two lists: (synthetic_samples, real_samples).
    
    """
    synthetic = []
    real = []

    for sample in samples:
        if sample.domain == "synthetic":
            synthetic.append(sample)

        if sample.domain == "real":
            real.append(sample)

    return synthetic, real
