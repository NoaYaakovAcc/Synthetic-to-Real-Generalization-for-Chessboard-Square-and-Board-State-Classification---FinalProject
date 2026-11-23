
"""
This file prepares the training, validation, and test datasets for the chessboard classification experiments.

The data preparation process is:
1. Collects labeled chessboard images from the selected game folders.
2. Filters images by domain (synthetic or real).
3. Groups images that share the same fen placement.
4. Split the images into training and internal validation sets, keeping identical FEN positions in the same split.
5. Ensure that the required data domains appear in both splits.
6. Keep the final real test games separate from training.
7. Create PyTorch DataLoaders for training and evaluation.

The file also supports loading previously saved data splits:
1. Read the image paths from splits.json.
2. Match the saved paths to the available labeled samples.
3. Check that images do not appear in multiple splits.
4. Verify the data domains and FEN separation.
5. Restore the training, validation, and test datasets.

Supported training modes are Zero-Shot, Fine-Tuning,
and Combined Training.
"""

import os
import random
import json
from pathlib import Path
from torch.utils.data import DataLoader

from config import SPLIT_ATTEMPTS
from data import scan_game, ChessBoardDataset
from pre_post_processing.preprocessing import ChessBoardTransform

def collect_samples(data_root, games):

    """
    Loads labeled board samples from all requested games.
    
    Parameters:
        data_root: Parent folder containing the game folders.
        games: game numbers to load.
    
    Returns:
        list[ChessBoardSample]: Combined list of ChessBoardSample objects.
    
    """

    samples = []

    for game in games:
        folder = os.path.join(data_root, "game" + str(game) + "_per_frame")
        csv_file = os.path.join(folder, "game" + str(game) + ".csv")

        if not os.path.isfile(csv_file):
            csv_file = os.path.join(folder, "positions.csv")

        game_samples = scan_game(folder, csv_file)
        samples.extend(game_samples)

    return samples

def filter_domain(samples, domain):

    """
    Keeps only images belonging to the data domain chosen.
    
    Parameters:
        samples: List of ChessBoardSample objects.
        domain:"synthetic" or "real".
    
    Returns:
        list[ChessBoardSample]: Filtered list of ChessBoardSample objects.
    
    """
    result = []

    for sample in samples:
        if sample.domain == domain:
            result.append(sample)

    return result

def fen_key(sample):

    """
    Gets the board position part of a fen string .
    
    Parameters:
        sample: ChessBoardSample containing a fen string.
    
    Returns:
        str: String describing only the piece positions, without move metadata.
    
    """
    sample = sample.fen.strip()
    return sample.split()[0]

def split_by_fen(samples, val_fraction, seed, required_domains):
    
    """
    Splits examples into train and internal validation 
    
    Parameters:
        samples: All images for this training mode.
        val_fraction: The fraction of images for internal validation.
        seed: Seed used.
        required_domains: Domains that must appear on both sides of the split.
    
    Returns:
        tuple[list[ChessBoardSample], list[ChessBoardSample]]: Tuple (train_samples, val_samples), with no FEN position shared between groups.
    
    """

    groups = {}

    for sample in samples:

        position = fen_key(sample)
        if position not in groups:
            groups[position] = []
        groups[position].append(sample)

    positions = list(groups.keys())
    if len(positions) < 2:
        raise ValueError("need at least two different fen positions")

    
    domain_totals = {}
    for domain in required_domains:
        domain_samples = filter_domain(samples, domain)
        domain_totals[domain] = len(domain_samples)

        unique_positions = set()
        for sample in domain_samples:
            unique_positions.add(fen_key(sample))

        if len(unique_positions) < 2:
            raise ValueError(domain + " needs at least two fen positions")

    random_generator = random.Random(seed)
    best_positions = None
    best_score = float("inf")

    # try several splits and choose one near the requested validation size.
    for attempt in range(SPLIT_ATTEMPTS):
        shuffled = positions.copy()
        random_generator.shuffle(shuffled)

        validation_size = round(len(positions) * val_fraction)
        validation_size = max(1, min(validation_size, len(positions) - 1))

        validation_positions = set(shuffled[:validation_size])

        validation_counts = {}
        for domain in required_domains:
            domain_count = 0

            for position in validation_positions:
                for sample in groups[position]:
                    if sample.domain == domain:
                        domain_count += 1

            validation_counts[domain] = domain_count

        valid_split = True

        for domain in required_domains:
            validation_count = validation_counts[domain]
            total_count = domain_totals[domain]

            if validation_count == 0 or validation_count == total_count:
                valid_split = False
                break

        if not valid_split:
            continue

        split_score = 0

        for domain in required_domains:
            actual_fraction = validation_counts[domain] / domain_totals[domain]
            difference = actual_fraction - val_fraction

            split_score += difference ** 2

        if split_score < best_score:
            best_score = split_score
            best_positions = validation_positions

    if best_positions is None:
        raise ValueError("Cannot create a train/validation split with both domains")

    # Separate samples into training and validation
    train_samples = []
    val_samples = []

    for sample in samples:
        position = fen_key(sample)

        if position in best_positions:
            val_samples.append(sample)
        else:
            train_samples.append(sample)

    return train_samples, val_samples

def build_loader(samples, train, batch_size, num_workers):
    
    """
    Creates a DataLoader for labeled chessboard images.

    Training samples are shuffled normally, without balancing
    synthetic and real examples within batches.
    """
    dataset = ChessBoardDataset(
        samples,
        transform=ChessBoardTransform(train=train)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers
    )

def print_split(name, samples):

    """
    Prints the sample counts and distinct fen positions in a split.
    
    Parameters:
        name: Label to print for this split.
        samples: List of samples in the split.
    
    Returns:
        None
        
    """
    synthetic_count = 0
    real_count = 0
    unique_positions = set()

    for sample in samples:
        if sample.domain == "synthetic":
            synthetic_count += 1
        if sample.domain == "real":
            real_count += 1
        unique_positions.add(fen_key(sample))

    print(name, "- total:", len(samples), "synthetic:", synthetic_count, "real:", real_count, "unique FEN:", len(unique_positions))

def prepare_samples(args):

    """
    Reads the data and divides it into training, validation and test sets.
    
    Parameters:
        args: Parsed settings like mode, game ids and split fraction.
    
    Returns:
        tuple[list[ChessBoardSample], list[ChessBoardSample], list[ChessBoardSample], str]: Tuple (train_samples, val_samples, test_samples, val_description).
    
    """
    synthetic = []
    real = []

    if args.mode in ["zero_shot", "combined"]:
        samples = collect_samples(args.data_root, args.synthetic_games)
        synthetic = filter_domain(samples, "synthetic")

    if args.mode in ["finetune", "combined"]:
        samples = collect_samples(args.data_root, args.real_games)
        real = filter_domain(samples, "real")

    test_samples = collect_samples(args.data_root, args.test_games)
    test_samples = filter_domain(test_samples, "real")

    if args.mode == "zero_shot":
        samples = synthetic
        required_domains = ["synthetic"]
        val_description = "synthetic validation"

    elif args.mode == "finetune":
        samples = real
        required_domains = ["real"]
        val_description = "real validation"

    else:
        samples = synthetic + real
        required_domains = ["synthetic", "real"]
        val_description = "combined validation"

    if len(samples) == 0:
        raise ValueError("No training images found")

    train_samples, val_samples = split_by_fen ( samples, args.val_fraction, args.seed, required_domains)

    print_split("TRAIN", train_samples)
    print_split("VALIDATION", val_samples)
    print_split("FINAL REAL TEST", test_samples)

    return train_samples, val_samples, test_samples, val_description


def prepare_samples_from_saved_split(args):
    """
    Loads a previously saved train, validation and test split.

    Parameters:
        args: Settings including split file, training mode and game IDs.

    Returns:
        train_samples, val_samples, test_samples, val_description
    """

    
    with open(args.splits_file, "r", encoding="utf-8") as file:
        saved_split = json.load(file)

    if saved_split["mode"] != args.mode:
        raise ValueError("Saved split mode does not match current mode")

    if sorted(saved_split["test_games"]) != sorted(args.test_games):
        raise ValueError("Saved test games do not match current test games")

    # collect samples
    synthetic_samples = []
    real_samples = []

    if args.mode in ["zero_shot", "combined"]:
        all_samples = collect_samples(args.data_root, args.synthetic_games)
        synthetic_samples = filter_domain(all_samples, "synthetic")

    if args.mode in ["finetune", "combined"]:
        all_samples = collect_samples(args.data_root, args.real_games)
        real_samples = filter_domain(all_samples, "real")

    test_samples = collect_samples(args.data_root, args.test_games)
    test_samples = filter_domain(test_samples, "real")

    # Select samples according to training mode
    if args.mode == "zero_shot":
        samples = synthetic_samples
        val_description = "synthetic validation"
        required_domains = ["synthetic"]

    elif args.mode == "finetune":
        samples = real_samples
        val_description = "real validation"
        required_domains = ["real"]

    else:
        samples = synthetic_samples + real_samples
        val_description = "combined validation"
        required_domains = ["synthetic", "real"]

    
    sample_map = {}

    for sample in samples + test_samples:
        image_path = normalize_path(sample.img_path)

        if image_path in sample_map:
            raise ValueError("Duplicate image path: " + image_path)

        sample_map[image_path] = sample

    
    used_paths = set()

    train_samples = restore_saved_group( "train", saved_split, sample_map, used_paths)
    val_samples = restore_saved_group( "epoch_validation", saved_split, sample_map, used_paths )
    final_test = restore_saved_group( "final_real_test", saved_split, sample_map, used_paths)

    # verifications that the split uploaded is valid

    for domain in required_domains:
        if not any(sample.domain == domain for sample in train_samples):
            raise ValueError("Missing training domain: " + domain)

        if not any(sample.domain == domain for sample in val_samples):
            raise ValueError("Missing validation domain: " + domain)

    for sample in train_samples + val_samples:
        if sample.domain not in required_domains:
            raise ValueError("Unexpected training or validation domain")

    for sample in final_test:
        if sample.domain != "real":
            raise ValueError("Test must contain real images only")

    # check for shared fen positions
    train_fens = {fen_key(sample) for sample in train_samples}
    val_fens = {fen_key(sample) for sample in val_samples}

    if train_fens.intersection(val_fens):
        raise ValueError("Train and validation share FEN positions")

    expected_test = {
        normalize_path(sample.img_path) for sample in test_samples
    }

    restored_test = {
        normalize_path(sample.img_path) for sample in final_test
    }

    if expected_test != restored_test:
        raise ValueError("Saved test set does not match current test images")

    print("Loaded saved split:", args.splits_file)

    print_split("TRAIN", train_samples)
    print_split("VALIDATION", val_samples)
    print_split("FINAL REAL TEST", final_test)

    return train_samples, val_samples, final_test, val_description

def normalize_path(path):

    """
    Normalizes a file path to a consistent format.

    Parameters:
        path: Path to a file or image

    Returns:
        str: Normalized file path

    """
    return os.path.normpath(str(path))


def restore_saved_group(group_name, saved_split, sample_map, used_paths):

    """
    Restores a group of samples from a previously saved split.

    Parameters:
        group_name: Name of the split to restore (train, epoch_validation, or final_real_test).
        saved_split: Saved split information loaded from splits.json.
        sample_map: Dictionary mapping image paths to their samples.
        used_paths: Image paths already assigned to other splits.

    Returns:
        list[ChessBoardSample]: Samples belonging to the restored group

    """

    paths = saved_split[group_name]

    if not isinstance(paths, list) or not paths:
        raise ValueError("Missing or empty split: " + group_name)

    group_samples = []

    for path in paths:
        image_path = normalize_path(path)

        if image_path in used_paths:
            raise ValueError("Image appears in multiple splits: " + image_path)

        if image_path not in sample_map:
            raise ValueError("Image not found in current dataset: " + image_path)

        used_paths.add(image_path)
        group_samples.append(sample_map[image_path])

    return group_samples
