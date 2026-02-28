import torch

class MySquareLoss:
    def __call__(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)