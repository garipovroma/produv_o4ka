import pandas as pd
import numpy as np
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, yaml.FullLoader)

files = config['files']

def read_target_train():
    return pd.read_csv(config['data_path'] + 'target_train.csv')

def read_data(suffix):
    data = {}
    for file in files:
        data[file] = pd.read_csv(f'{config["data_path"]}{file}_{suffix}.csv')
    data['chronom'].drop(['Unnamed: 0'], axis = 1, inplace = True)
    return data

def preprocess_data(data):
    X = data['chugun'].drop(['DATA_ZAMERA'], axis = 1)
    for file in files:
        if file == 'chugun':
            continue
        X = X.merge(data[file].groupby('NPLV').mean(), on = 'NPLV', suffixes=('',f'_{file}'))
    return X
