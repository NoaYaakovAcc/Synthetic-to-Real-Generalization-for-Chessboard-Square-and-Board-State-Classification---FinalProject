# Sim-to-Real Chessboard Square Classifier

This project uses a deep learning model based on **ConvNeXt-Tiny** to recognize chess pieces in chessboard images. It takes an **already cropped and aligned** chessboard image, extracts 64 overlapping square crops, and classifies each square into one of **13 classes**: 12 chess piece classes (six types in two colors) or empty.

The project studies the transfer from synthetic chessboard renders to real photographs using three training approaches:

- **Zero-Shot:** Train on synthetic images only.
- **Fine-Tuning:** Continue training a saved model checkpoint using real images.
- **Combined Training:** Train on synthetic and real images together.

The project also includes an **Integer Linear Programming (ILP)** decoder that applies chess-related constraints to the model's class scores. In the prediction script, ILP is the default method; Argmax remains available for comparison.

## 1. Environment setup

Run the following commands from a terminal. Training, evaluation, and inference were developed for a Python environment; synthetic image generation uses a **separate local Blender setup** described in Section 4.

### 1.1 Clone the repository

```bash
git clone <repository-url>
cd <repository-name>/code
```

Replace the placeholders with the actual GitHub repository URL and directory name. All training and inference commands below assume the working directory is `code/`.

### 1.2 Create and activate a Conda environment

```bash
conda create -n chess_project python=3.10 -y
conda activate chess_project
```

### 1.3 Install dependencies

```bash
python -m pip install -r requirements.txt
```

The project uses PyTorch, Torchvision, NumPy, Pandas, Pillow, OpenCV, Matplotlib, `python-chess`, and PuLP. The ILP component uses the **CBC** solver through PuLP; make sure CBC is available in the installed environment. For GPU execution, install a PyTorch build compatible with the system's CUDA driver.

### 1.4 Verify the installation

```bash
python -c "import torch, torchvision, cv2, chess, pulp; print('All imports OK'); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` is normal on systems without a compatible GPU. To reactivate the existing environment later, use `conda activate chess_project`.

## 2. Repository structure

The relevant source files are organized as follows (additional Blender-generation files and experiments may be stored separately):

```text
code/
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

- `config.py`: Default paths, hyperparameters, and class-label mappings.
- `data.py` and `utils/data_utils.py`: Scan labeled images, parse FEN labels, build loaders, and manage FEN-grouped data splits.
- `model/model.py`: ConvNeXt-Tiny classifier applied to 64 square crops.
- `pre_post_processing/preprocessing.py`: RGB resizing, normalization, and synthetic-training augmentation.
- `pre_post_processing/ilp_constraints.py`: ILP decoding with progressively stronger constraints.
- `train.py`, `utils/train_utils.py`, and `utils/run_utils.py`: Training and experiment management.
- `utils/eval_utils.py`: Metrics, confusion matrix, prediction examples, and Argmax–ILP comparisons.
- `predict/predict.py`: Single-image inference, FEN piece placement, and SVG visualization.

## 3. Datasets

### 3.1 Download the data

**The real and synthetic chessboard datasets are available in the shared Google Drive folder:**

**[Open the shared dataset folder on Google Drive](https://drive.google.com/drive/folders/1N6H6DVeqkYA5cxcQiJ63rsGe-8Z9u0Ka?hl=he)**

Download the required game folders and place them inside `code/data/`. Access to the folder may depend on its sharing permissions. Dataset files are distributed separately from the source-code repository.

### 3.2 Expected game-folder format

The code expects game folders named `game<number>_per_frame`, for example:

```text
data/
├── game5_per_frame/
│   ├── game5.csv
│   └── tagged_images/
│       └── frame_000001.jpg
├── game7_per_frame/
│   ├── game7.csv
│   └── tagged_images/
│       └── ...
└── game42_per_frame/
    ├── game42.csv
    ├── generated_images/
    │   └── frame_000616.jpg
    └── tagged_images/
        └── ...
```

Each game folder must contain either `game<number>.csv` or `positions.csv`. The dataset loader used here expects the columns:

```csv
from_frame,to_frame,fen
1,1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
2,2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
```

For the current loader, **`from_frame` and `to_frame` must match**. Image names must contain `frame` followed by the frame number, such as `frame_000616.jpg`. Images are matched to their labels by this number. Image paths containing `generated` are treated as **synthetic**, while paths containing `tagged` are treated as **real**.

### 3.3 Input format and classes

The input must show a **cropped, aligned chessboard**. Preprocessing converts it to RGB, resizes it to **480 × 480**, converts it to a PyTorch tensor, and normalizes it with ImageNet mean and standard deviation. Synthetic training images can additionally receive ColorJitter and GaussianBlur; real training, validation, and test images are not randomly augmented.

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

Predictions follow the **image's row order**: the first output element represents the top-left square of the supplied image. No fixed White/Black viewing orientation is assumed.

## 4. Generating new synthetic images with Blender (local computer)

The project has a separate Blender-based synthetic-data generator. **Use it on a local computer (the provided configuration targets Windows and Blender 5.0), not through the project's SLURM training setup.** Its rendering code was configured for NVIDIA OptiX; a different graphics card or CPU-only machine may require adapting the Blender device settings.

### 4.1 Files and installation

Prepare the following on the local machine:

1. **Blender 5.0**, installed locally.
2. The prepared chessboard scene, **`chess-set.blend`**.
3. The Blender positioning/rendering script, **`chess_position_api_v2.py`**.
4. The Python driver script containing **`generate_dataset()`** (use its actual filename when running it).
5. Python with **Pillow** installed for the driver script (`python -m pip install Pillow`). The Blender script uses Blender's bundled Python environment.
6. A folder containing a CSV with the FEN positions to render.

The `.blend` file needs the expected chessboard and piece objects. The provided script identifies pieces using object names and their initial positions, so an arbitrary `.blend` file may not work without changes.

### 4.2 Preparing FEN positions

Create a `.csv` file with one row for each image you want to generate:

```csv
from_frame,to_frame,fen
1,1,rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
2,2,rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
3,3,rnbqkbnr/pppppppp/8/8/4P3/5N2/PPPP1PPP/RNBQKB1R
```

The generator uses the **piece-placement portion of FEN** (eight ranks separated by `/`). Uppercase letters are white pieces; lowercase letters are black pieces. `K/Q/R/B/N/P` denote king, queen, rook, bishop, knight, and pawn. Digits `1`–`8` represent consecutive empty squares. Every rank must add up to eight squares. The `from_frame` value determines the output name, e.g. `1` becomes `frame_000001.jpg`; keep `to_frame` equal to `from_frame` for compatibility with the training loader.

The Blender script moves objects to the positions described by each FEN and can duplicate a piece object if the requested position has more pieces of that type than the original scene. It **does not certify chess legality**, so validate the CSV positions and inspect generated images before using them for training.

### 4.3 Configure local paths

At the beginning of the Python driver script, edit these absolute paths to match your computer:

```python
BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
BLEND_FILE = r"C:\path\to\chess-set.blend"
SCRIPT_FILE = r"C:\path\to\chess_position_api_v2.py"
RESOLUTION = 2400
```

Use **`blender.exe`**, not `blender-launcher.exe`. The provided driver uses absolute paths and calculates crop positions from `RESOLUTION`.

### 4.4 Run on Windows (PowerShell or terminal)

Navigate to the directory containing the driver script, then execute:

```powershell
python <dataset-generator-script>.py
```

Replace `<dataset-generator-script>` with the **actual filename** of the script that contains `generate_dataset()`; it is not the Blender script `chess_position_api_v2.py`.

The program prompts:

```text
Enter the directory path containing the CSV file:
```

Enter the folder containing the prepared `.csv`, for example:

```text
C:\Users\Owner\Documents\chess_data\game42_per_frame
```

The driver selects the **first `.csv` file** it finds in the directory. Keep only the intended CSV there to avoid using the wrong labels.

It then asks which rendered view to retain:

| Choice | View | Output crop |
|---|---|---|
| `1` or `overhead` | Overhead | Center of the render |
| `2` or `east` | East | Right-side crop |
| `3` or `west` | West | Left-side crop |

The default for an unrecognized entry is `overhead`. The current driver passes `--view white` and the selected `--view_angle` to Blender. **Only the selected view is rendered in each Blender invocation** with the supplied script; the driver then crops it and saves a JPEG image. The generated view naming and crop coordinates are specific to the provided scene and script.

For each row, the driver launches Blender in background mode, applies the FEN, renders, crops the output, and writes:

```text
game42_per_frame/
├── game42.csv
└── generated_images/
    ├── frame_000001.jpg
    ├── frame_000002.jpg
    └── ...
```

Existing output JPEGs are skipped. After successful cropping, temporary Blender PNG renders are deleted. Check the terminal output for rendering errors; failed frames may need to be rerun.

### 4.5 Use the generated images for training

Copy the completed `game<number>_per_frame` folder into `code/data/`. Keep the original CSV next to `generated_images/` so the loader can associate each image with its FEN. Select its game number through `--synthetic-games` during training.

**Note:** The local rendering workflow and SLURM training workflow are separate. You can download/upload generated images between the local computer, Google Drive, and the training environment; do not assume that Blender's OptiX setup will work on the SLURM nodes.

## 5. Training

Run these commands from `code/` with the Conda environment activated. The training script requires `--mode` and supports three modes.

### 5.1 Zero-shot (synthetic images only)

```bash
python train.py \
  --mode zero_shot \
  --synthetic-games 2 4 8 9 21 23 42 \
  --test-games 7 6 \
  --epochs 10
```

### 5.2 Fine-tuning (real images from a checkpoint)

```bash
python train.py \
  --mode finetune \
  --checkpoint path/to/best_model.pth \
  --real-games 5 \
  --test-games 7 6 \
  --epochs 10
```

By default, all model parameters are trainable. Optional alternatives are `--freeze-backbone` (train only the final classification layer) and `--partial-freeze` (freeze earlier ConvNeXt feature layers). These options are only supported in fine-tuning mode.

### 5.3 Combined training (synthetic + real)

```bash
python train.py \
  --mode combined \
  --synthetic-games 2 4 8 9 21 23 42 \
  --real-games 5 \
  --test-games 7 6 \
  --epochs 10
```

The game IDs above are the current defaults in `config.py`. Adjust them to the game folders you actually downloaded; test games must not also be selected for training.

### 5.4 Training arguments and selection

| Argument | Default | Meaning |
|---|---|---|
| `--epochs` | `10` | Number of epochs |
| `--batch-size` | `2` | Boards per batch |
| `--lr` | `0.0001` | Learning rate |
| `--optimizer` | `adamw` | `adamw` or `sgd` |
| `--val-fraction` | `0.20` | Requested internal validation fraction |
| `--seed` | `42` | Seed for reproducible splits |
| `--num-workers` | `4` | Data-loading workers |
| `--output-dir` | Automatic | Override experiment output directory |
| `--splits-file` | Not set | Restore a saved data split |
| `--evaluate-ilp` | Off | Also compare ILP and Argmax after training |

The current training code uses **ordinary Cross Entropy Loss without class weights**. AdamW is the default optimizer; `WEIGHT_DECAY` is configured in `config.py`. The selected checkpoint maximizes **validation non-empty accuracy**, breaking ties using validation loss.

### 5.5 Training, validation, and test splits

Training and internal validation are separated by **FEN piece-placement groups**: images with the same board position are kept together. The splitter tries several randomized group selections and prefers one close to the requested validation fraction for each required domain. Test games are selected separately from training games.

To reuse a saved split, pass `--splits-file` and the same mode and game settings:

```bash
python train.py \
  --mode combined \
  --synthetic-games 2 4 8 9 21 23 42 \
  --real-games 5 \
  --test-games 7 6 \
  --splits-file path/to/splits.json \
  --epochs 10
```

### 5.6 Training outputs

Each run creates an experiment directory with files such as:

```text
experiments/<experiment-name>/
├── models/
│   ├── best_model.pth
│   └── last_model.pth
├── curves/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── confusion_matrix.png
├── training_info/
│   ├── train_settings.json
│   ├── splits.json
│   ├── history.json
│   ├── final_metrics.json
│   └── per_class_metrics.csv
└── prediction_samples/
```

`best_model.pth` contains the best validation epoch's weights; `last_model.pth` contains the final epoch's weights. The evaluation and visualization files record measured behavior on the selected data.

## 6. Prediction (ILP or Argmax)

**Run from `code/`.** Because the prediction file lives in a `predict/` folder, use `python -m predict.predict` (no `.py` after the module name).

### 6.1 Select the checkpoint

The prediction function can use either the default path `MODEL_PATH` from `config.py` or the path supplied with `--checkpoint`.

For example, set `MODEL_PATH` in `config.py` to a real file:

```python
MODEL_PATH = "models/best_model.pth"
```

Or pass a different checkpoint when executing the command:

```bash
--checkpoint path/to/best_model.pth
```

The command-line checkpoint takes precedence. **For automatic course evaluation, `predict_board(image)` must work without a checkpoint argument, so the configured `MODEL_PATH` must point to a checkpoint that is available with the submission.**

### 6.2 Predict from an image

By default, the script uses **ILP** decoding with the `promotion` constraint level:

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

If the `--show-board` option is included in the current CLI implementation, add it to open the saved SVG with the local browser:

```bash
python -m predict.predict \
  --image path/to/chessboard.jpg \
  --checkpoint path/to/best_model.pth \
  --method ilp \
  --output predicted_board.svg \
  --show-board
```

On a remote SSH/SLURM session, a browser might not open on the local machine. The SVG is still saved and can be opened through VS Code or copied to the local computer.

The script prints the `(8, 8)` predicted tensor and **FEN piece placement**, and saves an SVG representation of the predicted board. Piece placement is only the first part of a full FEN; the model does **not** infer the active player, castling rights, or move counters.

### 6.3 Automatic evaluation API

The course evaluation calls:

```python
from predict.predict import predict_board

prediction = predict_board(image)
```

`image` must be an RGB NumPy array with shape `(H, W, 3)` and dtype `uint8`. The function returns a **CPU** `torch.Tensor` of shape `(8, 8)` and dtype `torch.int64` with class IDs `0`–`12`. The orientation follows the image (top-left to bottom-right). The default method is ILP, with an Argmax fallback if ILP solving fails.

## 7. ILP constraints and evaluation

The ILP decoder maximizes the total log probability of the chosen square classes and adds the following constraint levels in order:

| Method | Added constraints |
|---|---|
| `assignment` | Exactly one class per square |
| `kings` | Exactly one king of each color |
| `material` | Up to eight pawns and 16 total pieces per color |
| `adjacency` | Kings cannot be adjacent |
| `promotion` | Extra non-pawn pieces require missing pawns |

Each level includes the earlier levels. These rules improve structural consistency but do **not** guarantee that the board is a fully legal chess position.

To compare Argmax and ILP after a training run:

```bash
python train.py \
  --mode combined \
  --synthetic-games 2 4 8 9 21 23 42 \
  --real-games 5 \
  --test-games 7 6 \
  --evaluate-ilp
```

The comparison is saved as `training_info/ilp_results.json` and can include visual comparisons in `prediction_samples/`. It records square accuracy, occupied-square accuracy, perfect-board accuracy, corrected errors, introduced errors, and solver failures.

The main evaluation also calculates **square accuracy**, **non-empty accuracy**, **perfect-board accuracy**, and **macro F1**, and saves a confusion matrix and per-class metrics. Overall square accuracy should be interpreted alongside non-empty accuracy because many squares are empty.

## 8. Reproducing results

1. Install the dependencies from `requirements.txt`.
2. Download the data from the **[shared Google Drive folder](https://drive.google.com/drive/folders/1N6H6DVeqkYA5cxcQiJ63rsGe-8Z9u0Ka?hl=he)** and place the selected game folders under `code/data/`.
3. Read the experiment's `training_info/train_settings.json` for its mode, hyperparameters, game IDs, optimizer, and seed.
4. Reuse its `training_info/splits.json` using `--splits-file` when available.
5. Run `train.py` with the corresponding settings and compare the generated metrics and plots.
6. For inference, point `MODEL_PATH` or `--checkpoint` to the saved `best_model.pth`.

Minor differences are possible across library versions and hardware. For a complete reproduction of the best reported result, record the **specific experiment folder, checkpoint, command, and measured metrics** when publishing the final submission.

## 9. Limitations

The best observed square accuracy was approximately **87% in combined training** (the best-case experimental result, not a guaranteed result for every image). This is not sufficient to reliably recognize every square of an entire board. Performance remains sensitive to the synthetic-to-real domain gap.

The pipeline expects **already aligned, cropped chessboards** and does not perform automatic board localization or perspective correction. The Blender generator provides controlled aligned renders, but the generated images may differ from real photographs in materials, textures, lighting, shadows, and camera effects. ILP improves consistency only within its implemented constraints and is not a complete chess-legality checker.

Potential future work includes more varied synthetic rendering and domain randomization, a multi-stage occupancy/color/piece classifier, additional carefully validated chess constraints, and automatic board detection and perspective correction.
