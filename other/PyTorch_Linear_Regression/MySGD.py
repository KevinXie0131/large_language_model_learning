class MySGD:
    def __init__(self, parameters, lr=1e-2):
        self.lr = lr
        self.parameters = parameters

    def step(self):
        for name, param in self.parameters.items():
            param.data = param.data - self.lr * param.grad.data

    def zero_grad(self):
        for name, param in self.parameters.items():
            if param.grad is not None:
                param.grad.zero_()