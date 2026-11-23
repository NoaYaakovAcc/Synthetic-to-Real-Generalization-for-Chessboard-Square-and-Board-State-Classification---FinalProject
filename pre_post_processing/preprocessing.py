"""
This file prepares chessboard images before they are passed to the model.

The preprocessing flow is:
1. Converts each image to RGB.
2. Resizes the image to the required resolution.
3. Applies ColorJitter and GaussianBlur to synthetic training images.
4. Converts the image into a PyTorch tensor.
5. Normalizes the image using ImageNet mean and standard deviation.

Real training images and validation and test images do not receive augmentation.

"""
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from config import (
    RESOLUTION, IMAGENET_MEAN, IMAGENET_STD, COLOR_JITTER_PROB,
    BRIGHTNESS, CONTRAST, SATURATION, HUE,
    GAUSSIAN_BLUR_PROB, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SIGMA,
)


class ChessBoardTransform:

    """
    Prepares a chessboard image for ChessNet.
    Both real and synthetic images are assumed to be already cropped and aligned.

    Synthetic training images:
        Resize  -> ToTensor -> Normalize

    Real training images and all validation/test images:
        Resize -> ToTensor -> Normalize
    """

    def __init__(self, train=False):

        """
        Creates preprocessing and augmentation transformations.

        Parameters:
            train: Whether the images belong to the training set.

        Returns:
            none

        """

        self.train = train
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=BRIGHTNESS,
                    contrast=CONTRAST,
                    saturation=SATURATION,
                    hue=HUE,
                ),
            ], p=COLOR_JITTER_PROB),

            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=GAUSSIAN_BLUR_KERNEL,
                    sigma=GAUSSIAN_BLUR_SIGMA,
                ),
            ], p=GAUSSIAN_BLUR_PROB),
        ])

        # Standard preprocessing for all images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ])

    def __call__(self, image, domain="real"):
        """
        Resizes and preprocesses one chessboard image.

        Parameters:
            image: PIL image of an already aligned chessboard.
            domain: "real" or "synthetic".

        Returns:
            torch.Tensor: Normalized RGB image with shape (3, RESOLUTION, RESOLUTION).
        """

        if domain not in ("real", "synthetic"):
            raise ValueError(f"Unknown domain: {domain}")

        image_rgb = np.array(image.convert("RGB"))

        board = cv2.resize(
            image_rgb,
            (RESOLUTION, RESOLUTION),
            interpolation=cv2.INTER_LINEAR,
        )

        board = Image.fromarray(board)
        
        if self.train and domain == "synthetic" :
            board = self.augmentation(board)

        # Convert to tensor and normalize.
        return self.transform(board)


def preprocess_numpy_image(image_rgb, domain="real"):
    """
    Preprocesses a NumPy RGB image for inference.

    Parameters:
        image_rgb: NumPy RGB image with shape (H, W, 3).
        domain: "real" or "synthetic".

    Returns:
        torch.Tensor: Tensor with shape (1, 3, RESOLUTION, RESOLUTION).

    """

    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if image_rgb.dtype != np.uint8:
        raise TypeError("image must have dtype uint8")

    if domain not in ("real", "synthetic"):
        raise ValueError(f"Unknown domain: {domain}")

    board = cv2.resize(
        image_rgb,
        (RESOLUTION, RESOLUTION),
        interpolation=cv2.INTER_LINEAR,
    )

    image = Image.fromarray(board)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])

    return transform(image).unsqueeze(0)
