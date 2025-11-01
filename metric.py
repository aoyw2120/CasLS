import torch
import torch.nn as nn

EPSILON = 1e-7


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred = torch.clamp(pred, min=EPSILON, max=None)
        target = torch.clamp(target, min=EPSILON, max=None)
        return self.mse(torch.log2(pred + 1), torch.log2(target + 1))
