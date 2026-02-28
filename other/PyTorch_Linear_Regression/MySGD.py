class MySGD:
    def __init__(self, parameters, lr=1e-2):
        self.lr = lr
        self.parameters = parameters

    def step(self):
        for name, param in self.parameters.items():
            param.data = param.data - self.lr * param.grad.data / 16

    def zero_grad(self):
        for name, param in self.parameters.items():
            param.data.zero_()