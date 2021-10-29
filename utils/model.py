import pandas as pd
import numpy as np
import yaml
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

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


def metric(answers, user_csv):
    delta_c = np.abs(np.array(answers['C']) - np.array(user_csv['C']))
    hit_rate_c = np.int64(delta_c < 0.02)

    delta_t = np.abs(np.array(answers['TST']) - np.array(user_csv['TST']))
    hit_rate_t = np.int64(delta_t < 20)

    N = np.size(answers['C'])

    return np.sum(hit_rate_c + hit_rate_t) / 2 / N


def cross_val(X: pd.DataFrame, y: pd.DataFrame, model_tst, model_C, metric, n: int = 5) -> list:
    kf = KFold(n_splits=n)
    res = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_tst.fit(X_train, y_train['TST'])
        model_C.fit(X_train, y_train['C'].fillna(y_train['C'].mean()))

        pred_train = predict(X_train, model_tst, model_C)
        pred_test = predict(X_test, model_tst, model_C)

        res.append((metric(y_train, pred_train), metric(y_test, pred_test)))

    return res
