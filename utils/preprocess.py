import pandas as pd
import numpy as np
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, yaml.FullLoader)

files = config['files']

def read_target_train():
    target = pd.read_csv(config['data_path'] + 'target_train.csv')
    return target.fillna(target.mean())

def read_data(suffix):
    data = {}
    for file in files:
        data[file] = pd.read_csv(f'{config["data_path"]}{file}_{suffix}.csv')
    data['chronom'].drop(['Unnamed: 0'], axis = 1, inplace = True)
    return data
    
def preprocess_data(data):
    data['reduced_gas'] = ts_preproc(data['reduced_gas'], 'Time')
    data['reduced_produv'] = ts_preproc(data['reduced_produv'], 'SEC')
    X = data['chugun'].drop(['DATA_ZAMERA'], axis = 1)
    for file in files:
        if file == 'chugun':
            continue
        aggregated_data = data[file].groupby('NPLV').agg([np.mean, np.sum, np.min, np.max])
        aggregated_data.columns = list(map(lambda x: x[0] + "_" + x[1], aggregated_data.columns))
        X = X.merge(aggregated_data, on = 'NPLV', suffixes=('',f'_{file}'))
    return X

def ts_preproc(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    return df

def reduce_ts(ts_data:pd.DataFrame, chronom:pd.DataFrame) -> pd.DataFrame:
    new_ts_data = pd.DataFrame()
    for NPLV in ts_data.NPLV.unique():
        curr_new_ts = ts_data[ts_data['NPLV'] == NPLV][(ts_data[ts_data['NPLV'] == NPLV].index > 
                                                          (chronom[(chronom['NPLV'] == NPLV) & 
                                                                   (chronom['NOP'] == 'Продувка')]['VR_NACH'].values[0])) & 
                                                         (ts_data[ts_data['NPLV'] == NPLV].index < 
                                                          (chronom[(chronom['NPLV'] == NPLV) & 
                                                                   (chronom['NOP'] == 'Продувка')]['VR_KON'].values[0]))]
        new_ts_data = pd.concat((new_ts_data,curr_new_ts))
        
    return new_ts_data