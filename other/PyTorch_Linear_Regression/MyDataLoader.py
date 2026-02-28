import random

class MyDataLoader:

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        data_len = len(self.y)
        data_index = list(range(data_len))
        random.shuffle(data_index)
        batch_number = data_len // self.batch_size

        for idx in range(batch_number):
            start = idx * self.batch_size
            end = start + self.batch_size
            batch_indices = data_index[start:end]
            yield self.X[batch_indices], self.y[batch_indices]