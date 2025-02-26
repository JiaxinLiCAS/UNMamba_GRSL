import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from einops import rearrange


# SAD loss
class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_true, y_pred):
        if len(y_pred.shape) > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.view(-1, 1, y_true.shape[1])
        y_pred = y_pred.view(-1, 1, y_pred.shape[1])
        y_true_norm = torch.sqrt(torch.bmm(y_true, y_true.permute(0, 2, 1)))
        y_pred_norm = torch.sqrt(torch.bmm(y_pred, y_pred.permute(0, 2, 1)))
        summation = torch.bmm(y_pred, y_true.permute(0, 2, 1))
        angle = torch.acos(summation / (y_true_norm * y_pred_norm))
        sad = torch.mean(angle)
        return sad


class My_Loss(nn.Module):
    def __init__(self, weight_mse=1.0, weight_sad=1.0, weight_endm=0.001, weight_aban=1e-6):
        super(My_Loss, self).__init__()
        self.weight_mse = weight_mse
        self.weight_sad = weight_sad
        self.weight_endm = weight_endm
        self.weight_aban = weight_aban
        self.SAD = SADLoss()
        self.MSE = nn.MSELoss()

    def forward(self, y_true, y_pred, endm=None, hsi_mean=None, pred_aban=None):
        loss = 0
        if self.weight_mse != 0:
            loss += self.weight_mse * self.MSE(y_true, y_pred)

        if 1 < len(y_pred.shape) < 5:
            y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1).transpose(1, 2)
            y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1).transpose(1, 2)
        elif len(y_pred.shape) >= 5:
            y_true = rearrange(y_true, 'n b c w h -> (n b) (w h) c')
            y_pred = rearrange(y_pred, 'n b c w h -> (n b) (w h) c')

        if self.weight_sad != 0:
            loss_sad = self.weight_sad * self.SAD(y_true, y_pred)
            loss += loss_sad
        if endm is not None and hsi_mean is not None and self.weight_endm != 0:
            loss += self.weight_endm * self.MSE(hsi_mean, endm)
        if pred_aban is not None:
            aban_norm = torch.norm(pred_aban, p=0.5, dim=1)
            loss_aban = self.weight_aban * aban_norm.mean()
            loss += loss_aban
        return loss