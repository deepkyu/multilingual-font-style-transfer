import torch
import torch.nn as nn

class GANHingeLoss(nn.Module):
    def __init__(self):
        super(GANHingeLoss, self).__init__()
        self.relu = nn.ReLU()
        
    def hinge(self, pred: torch.Tensor, is_real=False) -> torch.Tensor:
        if is_real:
            return self.relu(1.0 - pred)
        return self.relu(1.0 + pred)
    
    def __call__(self, pred, is_real, for_discriminator):
        if for_discriminator:
            return (-1.0 * pred).mean()
        return self.hinge(pred, is_real).mean()