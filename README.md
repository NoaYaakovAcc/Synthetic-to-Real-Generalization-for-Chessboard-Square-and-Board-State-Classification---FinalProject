# Sim-to-Real Chessboard Square Classifier

This project uses a deep learning model with convnext-tiny backbone to recognize chess pieces in chessboard images. It takes a pre cropped and aligned chessboard image, extracts 64 expanded square crops and classifies each square into one of 13 classes: 12 chess piece classes (six types in two colors) or empty.

The project studies the transfer from synthetic chessboard images to real world images using three training methods:

- **Zero-shot:** Trains on synthetic images only.
- **Fine-tuning:** Continues training a saved pre trained model checkpoint using real images.
- **Combined training:** Trains on synthetic and real images together.

The project also includes an integer linear programming (ILP) decoder that applies chess related constraints to the model's class scores. In the prediction script, ILP is the default method but Argmax remains available for comparison if needed.

## 1. Environment setup

Run the following commands from a terminal. Training, evaluation, and inference were developed for a Python environment, synthetic image generation uses a eparate local Blender setup described in Section 4.

### 1.1 Clone the repository

```bash
git clone https://github.com/NoaYaakovAcc/Synthetic-to-Real-Generalization-for-Chessboard-Square-and-Board-State-Classification---FinalProject.git chessboard-project
cd chessboard-project
```

### 1.2 Create and activate a Conda environment

Choose either **Conda** or **venv**. Only one environment is needed. Python 3.10 is recommended.

**Option A: Conda**

```bash
conda create -n chess_classifier python=3.10 -y
conda activate chess_classifier
```

Make sure Python preferably 3.10 is installed, then create an environment from the project directory.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\Activate
```

Linux / WSL:

```bash
source .venv/bin/activate
```

After activating either environment, continue with Section 1.3. To leave it later, run `conda deactivate` for Conda or `deactivate` for venv. Add `.venv/` to `.gitignore` so the local environment is not committed to Git.
I personally prefer conda

### 1.3 Install dependencies

If you chose venv please start with:

```bash
python -m pip install --upgrade pip
```
Then for both choices:

```bash
python -m pip install -r requirements.txt
```

The project uses PyTorch, Torchvision, NumPy, Pandas, Pillow, OpenCV, Matplotlib, `python-chess`, and PuLP. The ILP component uses the **CBC** solver through PuLP; make sure CBC is available in the installed environment. For GPU execution, install a PyTorch build compatible with the system's cuda driver.

### 1.4 Verify the installation

```bash
python -c "import torch, torchvision, cv2, chess, pulp; print('All imports are good'); print('cuda available:', torch.cuda.is_available())"
```

## 2. Repository structure

The relevant source files are organized as follows (additional Blender-generation files and experiments are provided seperatelly in google drive):

```text
chessboard-project/
├── config.py
├── data.py
├── train.py
├── requirements.txt
├── model/
│   └── model.py
├── predict/
│   └── predict.py
├── pre_post_processing/
│   ├── preprocessing.py
│   └── ilp_constraints.py
├── utils/
│   ├── data_utils.py
│   ├── train_utils.py
│   ├── eval_utils.py
│   └── run_utils.py
├── data/
│   └── game*_per_frame/
└── experiments/
```

- `config.py`: Default paths, hyperparameters and class-label mappings.
- `data.py` and `utils/data_utils.py`: Scan labeled images, parse fen labels, build loaders and manage data splits.
- `model/model.py`: Convnext-tiny classifier applied to 64 square crops.
- `pre_post_processing/preprocessing.py`: rgb resizing, normalization and synthetic training augmentation.
- `pre_post_processing/ilp_constraints.py`: ILP decoding with constraints.
- `train.py`, `utils/train_utils.py`, and `utils/run_utils.py`: Training and experiment management.
- `utils/eval_utils.py`: Metrics, confusion matrix, prediction examples and Argmax–ILP comparisons.
- `predict/predict.py`: Single image inference, fen piece placement and svg visualization.

## 3. Datasets

### 3.1 Download the data

**The real and synthetic chessboard datasets are available in the shared Google Drive folder:**

**[Open the shared dataset folder on Google Drive](https://drive.google.com/drive/folders/1N6H6DVeqkYA5cxcQiJ63rsGe-8Z9u0Ka?hl=he)**

Download the required game folders and place them inside `data/`. Access to the folder may depend on its sharing permissions.

### 3.2 Expected game-folder format

The code expects game folders named `game<number>_per_frame`, for example:

```text
data/
├── game1_per_frame/
│   ├── game1.csv
│   └── tagged_images/
│       └── frame_000001.jpg
├── game2_per_frame/
│   ├── game2.csv
│   └── tagged_images/
│       └── ...
└── game3_per_frame/
    ├── game3.csv
    ├── generated_images/
    │   └── frame_000616.jpg
    └── tagged_images/
        └── ...
```

Each game folder must contain  `game<number>.csv` . The dataset loader used here expects the columns:

```csv
from_frame,to_frame,fen
1,1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
2,2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
```

For the current loader, `from_frame` and `to_frame` must match 
Image names must contain `frame` followed by the frame number, such as `frame_000616.jpg`. Images are matched to their labels by this number. For example `frame_000616.jpg` is mached with 616,616 fen.


Image paths containing `generated` are treated as synthetic, while paths containing `tagged` are treated as real.

### 3.3 Input format and classes

The input must show a cropped and aligned chessboard. Preprocessing converts it to rgb and resizes it.

The 13 output labels are:

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | White pawn | 6 | Black pawn |
| 1 | White rook | 7 | Black rook |
| 2 | White knight | 8 | Black knight |
| 3 | White bishop | 9 | Black bishop |
| 4 | White queen | 10 | Black queen |
| 5 | White king | 11 | Black king |
| 12 | Empty | | |

Predictions follow the image's row order: the first output element represents the top-left square of the supplied image.

## 4. Generating new synthetic images with blender 

The project includes a separate blender based pipeline for generating synthetic chessboard images from fen positions.
The generator was tested on Windows using Blender 5.0.

### 4.1 Files and installation

Prepare the following:

1. Blender 5.0 — Installed on the machine used for rendering.
2. chess-set.blend — The prepared blender scene containing the chessboard and chess pieces.
3. chess_position_api_v2.py — The blender script responsible for positioning pieces according to fen notation and rendering the board.
4. Dataset generation — The Python script containing generate_dataset(), which manages the rendering process and saves the generated images.
5. Python and Pillow — Required for the dataset generation. Install Pillow using:

```bash
python -m pip install Pillow
```
6. fen dataset csv — A csv file containing the chessboard positions to render.


### 4.2 Preparing FEN positions

Create a `.csv` file with one row for each image you want to generate:

```csv
from_frame,to_frame,fen
1,1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
2,2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
3,3,rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R
```

### 4.3 Configure local paths

At the beginning of the Python script, edit these absolute paths to match your computer:

```python
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = r"C:\path\to\chess-set.blend"
SCRIPT_FILE = r"C:\path\to\chess_position_api_v2.py"
RESOLUTION = 2400
```

Use `blender.exe`, not `blender-launcher.exe`. The provided script uses absolute paths.

### 4.4 Run on windows 

Navigate to the directory containing the driver script, then execute:

```powershell
python script_use_api.py
```

The program prompts:

```text
Enter the directory path containing the csv file (better to have in the same folder):
```

Enter the folder containing the prepared `.csv`, for example:

```text
C:\Users\Owner\Documents\chess_data
```

The script uses the first `csv` file found in the selected directory. Make sure it is the correct csv file.

Next, choose the camera view for generating the images:

| Choice | View | 
|---|---|
| `1`  | Overhead | 
| `2`  | East | 
| `3`  | West | 

The default view is Overhead.
For each fen position in the csv, the script runs blender, generates the selected view, crops the image and saves it as a `jpg` in the `generated_images`/ folder.

```text
└── generated_images/
    ├── frame_000001.jpg
    ├── frame_000002.jpg
    └── ...
```

Existing images are skipped, and temporary blender renders are removed after successful processing.

### 4.5 Use the generated images for training

Create a folder named `game<number>_per_frame` inside `data/`, replacing `<number>` with the game number.

Inside this folder, place the `generated_images/`, directory containing the generated images, and a CSV file named `game<number>.csv` containing the corresponding fen positions in the required format (from_frame,to_frame,fen).

The expected structure is:

```text
data/
└── game1_per_frame/
    ├── game1.csv
    └── generated_images/
        ├── frame_000001.jpg
        ├── frame_000002.jpg
        └── ...
```

To include the generated images in training, specify the corresponding game number using the --synthetic-games argument.

## 5. Training

Run these commands from `chessboard-project/` with the environment of your choice (conda or venv) activated. The training script requires `--mode` and supports three modes.


### 5.1 Zero-shot (synthetic images only)

```bash
python train.py \
  --mode zero_shot \
  --synthetic-games 1 2 3 \
  --test-games 5 6 \
  --epochs 10
```
For synthetic games 1 2 3 and final validation (test) on 5 6 for 10 epoches

### 5.2 Fine-tuning (real images + checkpoint)

```bash
python train.py \
  --mode finetune \
  --checkpoint path/to/best_model.pth \
  --real-games 4 \
  --test-games 5 6 \
  --epochs 10
```
For real game 4 and final validation (test) on 5 6 for 10 epoches

By default, all model parameters are trainable . 
Optional alternatives are `--freeze-backbone` (train only the final classification layer) and `--partial-freeze` (freeze earlier convnext feature layers). These options are only supported in fine-tuning mode.

### 5.3 Combined training (synthetic + real)

```bash
python train.py \
  --mode combined \
  --synthetic-games 1 2 3 \
  --real-games 4 \
  --test-games 5 6 \
  --epochs 10
```

For synthetic games 1 2 3, real game 4 and final validation (test) on 5 6 for 10 epoches

The game IDs above are the current defaults in `config.py`. Adjust them to the game folders you actually downloaded; test games must not also be selected for training.

### 5.4 Training arguments and selection

| Argument | Default | Meaning |
|---|---|---|
| `--epochs` | `10` | Number of epochs |
| `--batch-size` | `2` | Boards per batch |
| `--lr` | `0.0001` | Learning rate |
| `--optimizer` | `adamw` | `adamw` or `sgd` |
| `--val-fraction` | `0.20` | Requested internal validation fraction (ther rest for training) |
| `--seed` | `42` | Seed for reproducible splits |
| `--num-workers` | `4` | Data loading workers |
| `--output-dir` | In code | Override experiment output directory |
| `--splits-file` | Not set | Restore a saved data split |
| `--evaluate-ilp` | Off | Also compare ILP and Argmax after training |


### 5.5 Training, validation, and test splits

Training and internal validation are separated by fens: images with the same board position are kept together. The splitter tries several randomized group selections and prefers one close to the requested validation fraction for each required domain. 
To reuse a saved split, pass `--splits-file` and the same mode and game settings:

```bash
python train.py \
  --mode combined \
  --synthetic-games 1 2 3 \
  --real-games 4 \
  --test-games 5 6 \
  --splits-file path/to/splits.json \
  --epochs 10
```

### 5.6 Training outputs

Each run creates an experiment directory with files such as:

```text
experiments/<experiment-name>/
├── curves/
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── loss_curve.png
├── models/
│   ├── best_model.pth
│   └── last_model.pth
├── prediction_samples/
│   ├── prediction_00.png
│   ├── ...
│   ├── prediction_09.png
│   ├── ilp_comparison_00.png(If ilp chosen)
│   ├── ...
│   └── ilp_comparison_04.png
└── training_info/
    ├── final_metrics.json
    ├── history.json
    ├── ilp_results.json (If ilp chosen)
    ├── per_class_metrics.csv
    ├── splits.json
    └── train_settings.json
```

`best_model.pth` contains the best validation epoch's weights; `last_model.pth` contains the final epoch's weights. The evaluation and visualization files record measured behavior on the selected data.
 

## 6. Prediction (ILP or Argmax)

Run from `chessboard-project/`. 
Because the prediction file is in a `predict/` folder: 
```bash
python -m predict.predict
```
(no `.py` after the module name).

### 6.1 Select the checkpoint

The prediction function can use either the default path `MODEL_PATH` from `config.py` or the path supplied with `--checkpoint`.

For example, set `MODEL_PATH` in `config.py` to a real file:

```python
MODEL_PATH = "path/to/best_model.pth"
```

Or pass a different checkpoint when executing the command:

```bash
--checkpoint path/to/best_model.pth
```
The command-line checkpoint takes precedence. 
Per request an evaluation, `predict_board(image)` is available but the configured `MODEL_PATH`  must point to a model that is available.

### 6.2 Predict from an image

By default, the script uses ILP decoding:

```bash
python -m predict.predict \
  --image path/to/chessboard.jpg \
  --checkpoint path/to/best_model.pth \
  --method ilp \
  --output predicted_board.svg
```

For Argmax without ILP:

```bash
python -m predict.predict \
  --image path/to/chessboard.jpg \
  --checkpoint path/to/best_model.pth \
  --method argmax
```

### 6.3 Automatic evaluation API

The course evaluation calls:

```python
from predict.predict import predict_board

prediction = predict_board(image)
```

## 7. ILP constraints and evaluation

The ILP decoder maximizes the total log probability of the chosen square classes and adds the following constraint:

1. Exactly one class per square. 
2. Exactly one king of each color.
3. Up to eight pawns and 16 total pieces per color.
4. Kings cannot be adjacent.
5. Extra non-pawn pieces require missing pawns.

To compare Argmax and ILP after a training run:

```bash
python train.py \
  --mode combined \
  --synthetic-games 1 2 3 \
  --real-games 4 \
  --test-games 5 6 \
  --evaluate-ilp
```

The comparison is saved as `training_info/ilp_results.json` and can include visual comparisons in `prediction_samples/`.
It saves square accuracy, non-empty square accuracy, perfect-board accuracy, corrected errors, introduced errors and solver failures.

The main evaluation also calculates square accuracy, non-empty accuracy, perfect board accuracy, and macro F1  and saves a confusion matrix and per-class metrics. 

## 9. Limitations

The best observed square accuracy was approximately 87% in combined training (my best-case experimental result, not a guaranteed result for every image or experiment).
The pipeline expects already aligned and cropped chessboards and does not perform automatic board detection or perspective correction. The blender generator provides aligned renders.

