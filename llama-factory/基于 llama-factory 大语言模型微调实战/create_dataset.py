import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from jinja2 import Environment
template_path = r'C:\Users\KevinX\Downloads\weibo\template.txt'
senti_data_path = r'C:\Users\KevinX\Downloads\weibo\weibo_senti_100k.csv'
train_save_path = r'C:\Users\KevinX\Downloads\weibo\train_data'
def convert_to_json(data, data_type):
    # 读取模板
    env = Environment()
    env.policies['json.dumps_kwargs'] = {'ensure_ascii': False}
    template = env.from_string(open(template_path, encoding='utf8').read())
    datasets = []
    for idx, (label, review) in enumerate(data):
        review = review.replace('"', '')
        review = review.replace('\\', '')
        single = template.render({'input': review, 'output': label})
        if idx < len(data) - 1:
            single += ',\n'
        datasets.append(single)
    with open(f'{train_save_path}/{data_type}.json', 'w', encoding='utf8') as file:
        file.write('[\n')
        file.writelines(datasets)
        file.write('\n]')
def demo():
    reviews = pd.read_csv(senti_data_path)
    # 修改数据的标签
    reviews['label'] = np.where(reviews['label'] == 1, '好评', '差评')
    print('数据标签分布:', Counter(reviews['label']))
    # 删除某些长度评论
    reviews = reviews[reviews['review'].apply(lambda x: len(x) > 10 and len(x) < 300)]
    # 转换成列表格式
    reviews_list = reviews.to_numpy().tolist()
    # 原始数数据分割
    train_data, test_data = train_test_split(reviews_list,
                                             test_size=0.1,
                                             stratify=reviews['label'],
                                             random_state=345)
    print('全部训练集数量:', len(train_data))
    print('全部测试集数量:', len(test_data))
    # 控制微调的数据量
    train_data = train_data[:20000]
    # test_data = test_data[:5000]
    print('选择训练集数量:', len(train_data))
    print('选择测试集数量:', len(test_data))
    print('-' * 50)
    # 数据本地存储
    convert_to_json(train_data, 'train')
    convert_to_json(test_data,  'test')
def test():
    import torch
    # 如果输出为 True，则表示支持 CUDA
    print(torch.cuda.is_available())
    # 显示检测到的 GPU 数量
    print(torch.cuda.device_count())
    print(torch.__version__)
    # 验证 json 是否能够正确解析
    import json
    train_data = json.load(open(f'{train_save_path}/train.json', 'r', encoding='utf8'))
    print(train_data[0])
    test_data = json.load(open(f'{train_save_path}/test.json', 'r',   encoding='utf8'))
    print(test_data[0])
if __name__ == '__main__':
    demo()
    test()