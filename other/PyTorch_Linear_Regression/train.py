import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

import MyLinearRegression
import MySquareLoss
import MySGD
import MyDataLoader
import pickle


def train():
    # 加载数据
    data = pickle.load(open('train.pkl', 'rb'))
    # 训练参数
    epochs = 200
    estimator = MyLinearRegression.MyLinearRegression()
    optimizer = MySGD.MySGD(estimator.get_parameters(), lr=1e-3)
    criterion = MySquareLoss.MySquareLoss()

    epoch_loss = []
    for _ in range(epochs):
        total_loss = 0.0
        for train_x, y_true in MyDataLoader.MyDataLoader(data['data'], data['target'], 16):
            # 前向计算
            y_pred = estimator(train_x)
            # 梯度清零
            optimizer.zero_grad()
            # 损失计算
            loss = criterion(y_pred.reshape(-1), y_true)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 损失统计
            total_loss += loss.item() * len(y_true)
        epoch_loss.append(total_loss)

    plt.title('损失变化曲线')
    plt.plot(range(len(epoch_loss)), epoch_loss, linestyle='dashed')
    plt.grid()
    plt.show()

    # 存储模型
    pickle.dump(estimator, open('model.pkl', 'wb'))


if __name__ == '__main__':
    train()