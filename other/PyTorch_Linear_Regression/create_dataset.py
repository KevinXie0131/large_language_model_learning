import pickle
import numpy as np
from sklearn.datasets import make_regression
import torch


def create_dataset():
    X, y = make_regression(n_samples=256,
                           n_features=1,
                           noise=10,
                           random_state=0)

    # 打乱数据
    indices = np.arange(X.shape[0])  # 获取索引
    np.random.shuffle(indices)  # 打乱索引

    X_shuffled = X[indices]  # 打乱后的特征
    y_shuffled = y[indices]  # 打乱后的标签

    # 数据集分割
    train_size = 0.8
    train_number = int(X_shuffled.shape[0] * 0.8)
    X_train = X_shuffled[:train_number]
    y_train = y_shuffled[:train_number]

    X_test = X_shuffled[train_number:]
    y_test = y_shuffled[train_number:]

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float64)
    y_train = torch.tensor(y_train, dtype=torch.float64)

    X_test = torch.tensor(X_test, dtype=torch.float64)
    y_test = torch.tensor(y_test, dtype=torch.float64)

    # 存储数据
    pickle.dump({'data': X_train, 'target': y_train}, open('train.pkl', 'wb'))
    pickle.dump({'data': X_test, 'target': y_test}, open('test.pkl', 'wb'))


if __name__ == '__main__':
    create_dataset()