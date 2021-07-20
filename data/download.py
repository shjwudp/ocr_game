import pandas as pd
import json
import os
import urllib
# from urllib import request
import urllib.request

from pandas.io.parsers import read_csv
from tqdm import tqdm

urls = []

try:
    os.mkdir('./train_data/tianchi/image/')
except:
    pass

# 训练集
train = pd.concat([
    pd.read_csv('input/Xeon1OCR_round1_train1_20210526.csv'),
    pd.read_csv('input/Xeon1OCR_round1_train2_20210526.csv'),
    pd.read_csv('input/Xeon1OCR_round1_train_20210524.csv')]
)

for row in train.iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    urls.append(path)

# 测试集
train = pd.concat([
    pd.read_csv('input/Xeon1OCR_round1_test1_20210528.csv'),
    pd.read_csv('input/Xeon1OCR_round1_test2_20210528.csv'),
    pd.read_csv('input/Xeon1OCR_round1_test3_20210528.csv')]
)

for row in train.iterrows():
    path = json.loads(row[1]['原始数据'])['tfspath']
    urls.append(path)

print('Total images: ', len(urls))

def down_image(url):
    print(url)
    if os.path.exists('./train_data/tianchi/image/' + url.split('/')[-1]):
        return
    urllib.request.urlretrieve(path, './train_data/tianchi/image/' + url.split('/')[-1])

from joblib import Parallel, delayed
Parallel(n_jobs=-1)(delayed(down_image)(url) for url in tqdm(urls))
