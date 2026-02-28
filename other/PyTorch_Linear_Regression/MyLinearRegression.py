import torch

class MyLinearRegression:

    def __init__(self):
        self.w = torch.tensor(0.1, requires_grad=True, dtype=torch.float64)
        self.b = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)

    def __call__(self, x):
        return self.w * x + self.b

    def get_parameters(self):
        return {'w': self.w, 'b': self.b}

    def eval(self):
        self.w.requires_grad = False
        self.b.requires_grad = False