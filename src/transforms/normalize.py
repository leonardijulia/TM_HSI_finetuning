import torch
import torch.nn as nn

from torch import Tensor



class NormalizeMeanStd(nn.Module):
    """Normalization module for hyperspectral data standardization.
    
    This module performs channel-wise standardization of input tensors
    using precomputed mean and standard deviation statistics. 
    
    The module is compatible with Kornia and PyTorch Lightning augmentation pipelines.
    """
    indices = {"enmap": [6, 16, 30, 48, 54, 59, 65, 71, 75, 90, 131, 172],
               "hyperview_1": [0, 9, 30, 63, 76, 87, 101, 116, 126, 149, 149, 149]}
    
    def __init__(self, 
                 mean: Tensor, 
                 std: Tensor, 
                 indices: str = "enmap"):
        """Initialize the normalization module with precomputed statistics.
        
        Args:
            mean (Tensor): A 1D tensor containing the mean value for each spectral band.
                Should have shape [num_channels].
            std (Tensor): A 1D tensor containing the standard deviation for each spectral band.
                Should have shape [num_channels].
            indices (str): Key to select which set of band indices to use for normalization.
                Must be one of the keys in the `indices` dictionary.
        """
        super().__init__()
        
        assert indices in self.indices.keys(), f"Invalid indices set '{indices}'. Must be one of {list(self.indices.keys())}"
        self.mean = mean[self.indices[indices]]
        self.std = std[self.indices[indices]]


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
        
        x_out = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)

        return x_out
