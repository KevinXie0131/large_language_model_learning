import MyLinearRegression
import MySquareLoss
import MySGD
import MyDataLoader
import torch
import pickle


def eval():
    # 加载数据
    data = pickle.load(open('test.pkl', 'rb'))
    y_true = data['target']
    estimator = pickle.load(open('model.pkl', 'rb'))
    estimator.eval()
    print(estimator.get_parameters())
    y_pred = estimator(data['data'])
    mse = MySquareLoss.MySquareLoss()(y_pred.reshape(-1), y_true)
    print('MSE:', mse)


if __name__ == '__main__':
    eval()