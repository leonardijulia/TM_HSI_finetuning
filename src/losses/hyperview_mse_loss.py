import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomMSELoss(nn.Module):
    """This is a custom MSELoss from the Hyperview challenge."""
    
    def __init__(self, baseline_outputs=None):
        super().__init__()
        if baseline_outputs is None:
            # calculated mean of the training dataset
            baseline_outputs = torch.tensor([4.13706118e-08, 1.21617560e-07, -1.18671728e-07, 1.23106165e-05]) 
        self.register_buffer("baseline_outputs", baseline_outputs)

    def forward(self, y_pred, y_true):
        baseline_outputs = self.baseline_outputs.to(y_true.device)
        mse_model = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
        
        baseline_tensor = baseline_outputs.unsqueeze(0).expand_as(y_true)
        mse_baseline = F.mse_loss(baseline_tensor, y_true, reduction="none").mean(dim=0)
        
        # Prevent division by zero
        mse_baseline = torch.clamp(mse_baseline, min=0.1)
        normalized_mse = mse_model / mse_baseline
        
        return normalized_mse.mean()