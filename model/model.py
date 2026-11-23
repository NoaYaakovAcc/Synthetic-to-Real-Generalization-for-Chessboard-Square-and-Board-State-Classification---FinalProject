"""
This file defines ChessNet, the chessboard square classification model.

The prediction process is:
1. Receives a batch of aligned chessboard images.
2. Divides each board into 64 square crops.
3. Enlarges the crops.
4. Passes all square crops through ConvNeXt-Tiny.
5. Predicts scores for 13 classes for each square.
6. Reshapes the predictions back into 64 squares per board.

The model returns a tensor with shape (batch_size, 64, 13).

"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CLASSES, RESOLUTION, BOARD_SIZE, EXPANSION_RATIO


class ChessNet(nn.Module):

    """
    Classifies all 64 squares of a chessboard with ConvNeXt Tiny.

    Each input board is converted into 64 crops. 
    
    Returns one set of class scores per square.

    """

    def __init__(self, num_classes=NUM_CLASSES, resolution=RESOLUTION, expansion_ratio=EXPANSION_RATIO, pretrained=True):

        
        """
        Build ConvNeXt Tiny and set up square crop dimensions.
        
        Parameters:
            num_classes: Number of classes predicted for each square.
            resolution: Width and height of the input chessboard.
            expansion_ratio: Amount by which each square crop is enlarged.
            pretrained: If loading pretrained ImageNet weights is needed
        
        Returns:
            none
            
        """
        super().__init__()

        self.resolution = resolution
        self.base_tile = resolution // BOARD_SIZE


        self.expanded_tile = round(self.base_tile * expansion_ratio)

        # ensures the larger crop can be centered on each square when padded it
        if self.expanded_tile % 2 != self.base_tile % 2:
            self.expanded_tile += 1

        self.pad = max(0, (self.expanded_tile - self.base_tile) // 2)

        
        if pretrained:
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.convnext_tiny(weights=weights)

        # replaces the final layer so it predicts 13 chess square classes instead of 768
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(num_features,num_classes)


        

    def _extract_tiles(self, images):
        
        """
        Splits the input boards into 64 squares each.
        
        Parameters:
            images: Batch of RGB tensors with shape (B, 3, H, W).
        
        Returns:
            torch.Tensor: Tensor of squares with shape (B * 64, 3, expanded_tile, expanded_tile).
            
        """

        height = images.shape[2]
        width = images.shape[3]

        if height != self.resolution or width != self.resolution:
            images = F.interpolate(
                images,
                size=(self.resolution, self.resolution),
                mode="bilinear",
                align_corners=False
            )

        # adds a border for crop
        if self.pad > 0:
            images = F.pad(
                images,
                (self.pad, self.pad, self.pad, self.pad),
                mode="replicate"
            )

        tiles = images.unfold(2, self.expanded_tile, self.base_tile)
        tiles = tiles.unfold(3, self.expanded_tile, self.base_tile)

        # sets the dimensions in the order: board, row, column, color, height, width
        tiles = tiles.permute(0, 2, 3, 1, 4, 5)
        tiles = tiles.contiguous()

        tiles = tiles.reshape(-1, 3, self.expanded_tile, self.expanded_tile)
        return tiles

    def forward(self, images):

        """
        Predicts a class score for each of the 64 chessboard squares.
        
        Parameters:
            images: Batch of board tensors with shape (B, 3, H, W).
        
        Returns:
            torch.Tensor: Tensor with shape (B, 64, num_classes).
        
        """

        batch_size = images.shape[0]

        # splits each board
        tiles = self._extract_tiles(images)
        
        # classifying
        predictions = self.backbone(tiles)

        # groups the predictions back
        predictions = predictions.reshape(batch_size, BOARD_SIZE * BOARD_SIZE, -1)

        return predictions
