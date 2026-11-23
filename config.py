# Board and classes
RESOLUTION = 480
BOARD_SIZE = 8
BOARD_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_CLASSES = 13
EMPTY_CLASS = 12
EXPANSION_RATIO = 1.4

# Training defaults
BATCH_SIZE = 2
NUM_WORKERS = 4
SEED = 42
EPOCHS = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
VAL_FRACTION = 0.20
SPLIT_ATTEMPTS = 1200

DATA_ROOT = "data"
SYNTHETIC_GAMES = [2, 4, 8, 9, 21, 23, 42]
REAL_GAMES = [5]
TEST_GAMES = [7 , 6]

MODEL_PATH = "/home/noayaak/own_final_project/new_final_project/code/experiments/ablation/combined_s248_21_23_42_9_r5_adamw_7/models/best_model.pth"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ColorJitter
COLOR_JITTER_PROB = 0.8
BRIGHTNESS = 0.15
CONTRAST = 0.15
SATURATION = 0.10
HUE = 0.02

# GaussianBlur
GAUSSIAN_BLUR_PROB = 0.2
GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA = (0.1, 0.8)

# Weighted loss
PIECE_LOSS_WEIGHT = 1.25
EMPTY_LOSS_WEIGHT = 0.75

# Evaluation
NUM_VISUALIZATIONS = 10

# Class labels: same ordering as the model output and fen  labels

PIECE_TO_INDEX = {
    "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5,
    "p": 6, "r": 7, "n": 8, "b": 9, "q": 10, "k": 11,
    ".": EMPTY_CLASS
}

CLASS_NAMES = [
    "WP", "WR", "WN", "WB", "WQ", "WK",
    "BP", "BR", "BN", "BB", "BQ", "BK", "Empty"
]

IDX_TO_PIECE = {
    0: "P", 1: "R", 2: "N", 3: "B", 4: "Q", 5: "K",
    6: "p", 7: "r", 8: "n", 9: "b", 10: "q", 11: "k",
    EMPTY_CLASS: "."
}

FULL_CLASS_NAMES = [
    "White Pawn", "White Rook", "White Knight", "White Bishop",
    "White Queen", "White King", "Black Pawn", "Black Rook",
    "Black Knight", "Black Bishop", "Black Queen", "Black King", "Empty"
]

FEN_TO_UNICODE = {
    "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
    "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
    ".": ""
}
