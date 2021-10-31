import pandas as pd
import numpy as np
import yaml
import tsfel
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

def chronom_preprocess_cat_features(chronom, X):
    dict_1 = chronom["NOP"].value_counts().to_dict()
    ok_col_values = {'Вхождение в гр.МНЛЗ', 'Завалка лома', 'Заливка чугуна', 'Замена фурмы', 'Замер положения фурм',
     'Наведение гарнисажа', 'Наложение продувки', 'Неиспр. АСУ и КИПиА', 'Неиспр. механ. обор.', 'Неиспр. электр. обор',
     'Неиспр. энерг. обор', 'Обрыв горловины', 'Ожидание стальковша', 'Ожидание шл.чаш', 'Осмотр конвертера', 
                     'Отсут. своб.разл.пл.', 'Отсутствие O2', 'Отсутствие мет.шихты', 'Отсутствие чугуна', 'ППР', 
                     'Подварка  футеровки', 'Полусухое торкрет.', 'Продувка', 'Ремонт летки'}
    for k, v in dict_1.items():
        if k in ok_col_values:
            temp = chronom.groupby('NPLV').apply(lambda grp: np.any(grp['NOP'] == k)).to_frame()
            temp.columns = [k + "_gr"]
            X = X.merge(temp, on='NPLV', suffixes='')
    return X    

def get_cat_features_list(df):
    result = []
    selected = list(df.select_dtypes(['bool', 'object']))
    result += selected
    result.append('labels')
    return result.copy()

def preprocess_data(data):
    X = data['chugun'].drop(['DATA_ZAMERA'], axis = 1)
    for file in files:
        if file in ['chugun', 'plavki']:
            continue
        aggregated_data = data[file].groupby('NPLV').agg([np.mean, np.sum, np.min, np.max, np.median, 'last', 'first', 'idxmax', 'idxmin'])
        aggregated_data.columns = list(map(lambda x: x[0] + "_" + x[1], aggregated_data.columns))
        X = X.merge(aggregated_data, on = 'NPLV', suffixes=('',f'_{file}'))
        
    X = chronom_preprocess_cat_features(data['chronom'], X)
    
#     data['reduced_gas'] = ts_preproc(data['reduced_gas'], 'Time')
#     data['reduced_produv'] = ts_preproc(data['reduced_produv'], 'SEC')
#     data['extracted_gas'] = ts_extract_features(data['reduced_gas'])
#     data['extracted_produv'] = ts_extract_features(data['reduced_produv'])
    
#     X = X.merge(data['extracted_gas'], on = 'NPLV', suffixes=('',f'_extracted_gas'))
#     X = X.merge(data['extracted_produv'], on = 'NPLV', suffixes=('',f'_extracted_produv'))
    X = X.merge(data['cat_lom'], on='NPLV', suffixes=('', f'_cat_lom'))
    X = X.merge(data['cat_sip'], on='NPLV', suffixes=('', f'_cat_sip'))
    X = X.merge(data['cat_sip'], on='NPLV', suffixes=('', f'_cat_sip'))


    return X

def read_preprocessed_data():
    return pd.read_csv(config['data_path'] + 'preprocessed_train.csv'), pd.read_csv(config['data_path'] + 'preprocessed_test.csv')
    
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

def ts_extract_features(ts_df:pd.DataFrame) -> pd.DataFrame:
    header_names = ts_df.drop(['NPLV'], axis = 1).columns
    cfg_file = tsfel.get_features_by_domain()
    extr_ts = pd.DataFrame()
    for NPLV in tqdm(ts_df.NPLV.unique()):
        curr_ts = ts_df[ts_df['NPLV'] == NPLV]
        curr_ts = curr_ts.drop(['NPLV'], axis = 1)

        feat_for_cts = tsfel.time_series_features_extractor(cfg_file, curr_ts, header_names = header_names, verbose=0)
        feat_for_cts['NPLV'] = NPLV

        extr_ts = pd.concat((extr_ts,feat_for_cts))
    
    return extr_ts

def ts_select_features(feat_df_train:pd.DataFrame, feat_df_test:pd.DataFrame) -> pd.DataFrame:
    corr_features = tsfel.correlated_features(feat_df_train)
    feat_df_train.drop(corr_features, axis=1, inplace=True)
    feat_df_test.drop(corr_features, axis=1, inplace=True)
    
    selector = VarianceThreshold()
    selector.fit(feat_df_train)
    new_features = selector.get_feature_names_out()
    
    return feat_df_train[new_features], feat_df_test[new_features]


def preprocess_cat_features():
    X = train_data['chronom']
    for file in files:
        cols = list(train_data[file].select_dtypes(['object', 'string']).columns)

        cols.append('NPLV')
        print(file, cols)