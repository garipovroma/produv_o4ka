import pandas as pd
import numpy as np
import yaml
from catboost import CatBoostRegressor

with open('config.yaml') as f:
    config = yaml.load(f, yaml.FullLoader)

def save_catboost_models(model_tst, model_C):
    model_tst.save_model(config['model_tst_path'])
    model_C.save_model(config['model_C_path'])
    
def get_catboost_models():
    model_tst = CatBoostRegressor()
    model_tst.load_model(config['model_tst_path'])
    model_C = CatBoostRegressor()
    model_C.load_model(config['model_C_path'])
    return model_tst, model_C

def predict(test_data, model_tst, model_C):
    y_pred = test_data[['NPLV']]
    y_pred['TST'] = model_tst.predict(test_data)
    y_pred['C'] = model_C.predict(test_data)
    return y_pred

def save_to_csv(y):
    y.to_csv(config['submission_path'])