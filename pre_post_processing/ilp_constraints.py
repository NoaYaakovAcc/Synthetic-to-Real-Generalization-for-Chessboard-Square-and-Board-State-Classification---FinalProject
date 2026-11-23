"""
This file applies chess-related constraints to model predictions using ILP.

The ILP process is:
1. Receives the model's class scores for all 64 squares.
2. Converts the scores into log probabilities.
3. Create binary variables for assigning classes to squares.
4. Maximizes the total log probability of the selected classes.
5. Applies chess constraints according to the selected method.
6. Solves the optimization problem and return 64 class predictions.

The constraints are added gradually
The purpose here is to improve the consistency of predicted chessboards.

"""
import numpy as np
import pulp
import torch

from config import NUM_CLASSES, PIECE_TO_INDEX, BOARD_SQUARES


METHODS = ["assignment", "kings", "material", "adjacency", "promotion"]


def solve_board(logits, time_limit=15, method="promotion"):

    """
    Predict the class of each chessboard square using ILP.

    Each method adds constraints to the previous one:
        assignment: One class per square
        kings: Exactly one king of each color
        material: Limits on pawns and total pieces
        adjacency: Kings cannot be next to each other
        promotion: Extra pieces require missing pawns

    Args:
        logits: Model scores with shape (64, 13).
        time_limit: Maximum solving time in seconds.
        method: Which constraints to apply.

    Returns:
        NumPy array containing 64 predicted class indices.

    """

    if method not in METHODS:
        raise ValueError(f"Unknown method")

    if tuple(logits.shape) != (BOARD_SQUARES, NUM_CLASSES):
        raise ValueError("Invalid logits shape")

    method_level = METHODS.index(method)

    probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities =probabilities.detach()
    probabilities = probabilities.cpu().numpy()


    problem = pulp.LpProblem("ilp_constraints", pulp.LpMaximize)

    selected_classes = pulp.LpVariable.dicts(
        "selected",
        (range(BOARD_SQUARES), range(NUM_CLASSES)),
        cat="Binary"
    )

    problem += pulp.lpSum(
        float(probabilities[square, piece_class])
        * selected_classes[square][piece_class]
        for square in range(BOARD_SQUARES)
        for piece_class in range(NUM_CLASSES)
    )

    for square in range(BOARD_SQUARES):
        problem += pulp.lpSum(
            selected_classes[square][piece_class]
            for piece_class in range(NUM_CLASSES)
        ) == 1

    if method_level >= 1:
        for king in ("K", "k"):
            king_class = PIECE_TO_INDEX[king]
            problem += pulp.lpSum(
                selected_classes[square][king_class]
                for square in range(BOARD_SQUARES)
            ) == 1

    if method_level >= 2:
        for pawn, pieces in [ ("P", "PRNBQK"), ("p", "prnbqk")]:
            problem += pulp.lpSum(
                selected_classes[square][PIECE_TO_INDEX[pawn]]
                for square in range(BOARD_SQUARES)
            ) <= 8

            problem += pulp.lpSum(
                selected_classes[square][PIECE_TO_INDEX[piece]]
                for square in range(BOARD_SQUARES)
                for piece in pieces
            ) <= 16

   
    if method_level >= 3:
        white_king = PIECE_TO_INDEX["K"]
        black_king = PIECE_TO_INDEX["k"]

        for row in range(8):
            for col in range(8):
                white_square = row * 8 + col

                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        neighbor_row = row + row_offset
                        neighbor_col = col + col_offset

                        if not (0 <= neighbor_row < 8 and
                                0 <= neighbor_col < 8):
                            continue

                        black_square = neighbor_row * 8 + neighbor_col

                        problem += ( selected_classes[white_square][white_king] + selected_classes[black_square][black_king] <= 1)

    if method_level >= 4:
        for pawn, starting_pieces in [
            ("P", [("Q", 1), ("R", 2), ("B", 2), ("N", 2)]),
            ("p", [("q", 1), ("r", 2), ("b", 2), ("n", 2)])
        ]:
            extra_pieces = []

            for piece, starting_count in starting_pieces:
                extra_count = pulp.LpVariable(
                    f"extra_{piece}",
                    lowBound=0
                )

                extra_pieces.append(extra_count)

                problem += pulp.lpSum(
                    selected_classes[square][PIECE_TO_INDEX[piece]]
                    for square in range(BOARD_SQUARES)
                ) <= starting_count + extra_count

            pawn_count = pulp.lpSum(
                selected_classes[square][PIECE_TO_INDEX[pawn]]
                for square in range(BOARD_SQUARES)
            )

            problem += pawn_count + pulp.lpSum(extra_pieces) <= 8

    # solve the optimization problem.
    solver = pulp.PULP_CBC_CMD(
        msg=False,
        timeLimit=time_limit
    )

    problem.solve(solver)

    if pulp.LpStatus[problem.status] != "Optimal":
        raise RuntimeError(
            f"ILP solver failed: {pulp.LpStatus[problem.status]}"
        )

    
    predictions = []

    for square in range(BOARD_SQUARES):
        chosen_classes = [
            piece_class
            for piece_class in range(NUM_CLASSES) if pulp.value(selected_classes[square][piece_class]) > 0.5
        ]

        if len(chosen_classes) != 1:
            raise RuntimeError(f"Invalid prediction for square {square}")

        predictions.append(chosen_classes[0])

    return np.array(predictions, dtype=np.int64)
