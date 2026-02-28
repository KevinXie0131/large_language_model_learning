import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def show_info(data):

    review_sizes = []
    for label, review in data.to_numpy().tolist():
        review_sizes.append(len(review))

    print('最大长度:', max(review_sizes))
    print('最小长度:', min(review_sizes))
    print('平均长度:', int(sum(review_sizes) / len(review_sizes)))
    print('-' * 50)


def demo():
    # data = pd.read_csv('ChnSentiCorp_htl_8k/ChnSentiCorp_htl_8k.csv')
    data = pd.read_csv('weibo_senti_100k/weibo_senti_100k.csv', encoding='utf-8')
    data['label'] = np.where(data['label'] == 1, '好评', '差评')

    print('数据标签分布:', Counter(data['label']))
    print('-' * 50)

    # 去掉太长的评论
    data = data[data['review'].apply(lambda x: len(x) > 10 and len(x) < 300)]
    show_info(data)

    # 原始数数据分割
    train_data, test_data  = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)

    print('原始训练集数量:', train_data.shape)
    print('原始测试集数量:', test_data.shape)
    print('-' * 50)

    # 采样部分数据
    sample_num = 5000
    train_data = train_data.sample(int(sample_num * 0.8), random_state=42)
    test_data  = test_data.sample(int(sample_num * 0.2),  random_state=52)

    print('最终训练集数量:', train_data.shape)
    print('最终测试集数量:', test_data.shape)

    # 数据转换字典
    train_data = train_data.to_dict(orient='records')
    test_data  = test_data.to_dict(orient='records')

    # 数据本地存储
    pickle.dump(train_data, open('weibo_senti_100k/01-训练集.pkl', 'wb'))
    pickle.dump(test_data,  open('weibo_senti_100k/02-测试集.pkl', 'wb'))


if __name__ == '__main__':
    demo()