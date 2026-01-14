import torch
import torch.nn as nn
import numpy as np

from torch import Tensor

class NormalizeMeanStd(nn.Module):
    """Normalization module for hyperspectral data standardization.
    
    This module performs channel-wise standardization of input tensors
    using precomputed mean and standard deviation statistics. 
    
    The module is compatible with Kornia and PyTorch Lightning augmentation pipelines.
    """
    
    def __init__(self, 
                 mean: Tensor, 
                 std: Tensor):
        """Initialize the normalization module with precomputed statistics.
        
        Args:
            mean (Tensor): A 1D tensor containing the mean value for each spectral band.
                Should have shape [num_channels].
            std (Tensor): A 1D tensor containing the standard deviation for each spectral band.
                Should have shape [num_channels].
        """
        super().__init__()
        
        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))


    @torch.no_grad()  
    def forward(self, x):
        """Normalize the input tensor using the stored mean and standard deviation.
        
        The normalization is applied channel-wise, using the formula:
        x_normalized = (x - mean) / std
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, height, width].
                
        Returns:
            Tensor: Normalized tensor with the same shape as the input.
        """
        
       return (x - self.mean) / self.std
