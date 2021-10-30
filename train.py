import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import *
from utils.model import *

# Read data
print("reading target")
y = read_target_train()
print("reading train data")
train_data = read_data('train')
print("reading test data")
test_data = read_data('test')
print("preprocess data")
X = preprocess_data(train_data)



# Train catboost

from catboost import CatBoostRegressor
print("train catboost")
model_tst = CatBoostRegressor(verbose = 100, random_state=42)
model_C = CatBoostRegressor(verbose = 100, random_state=42)

model_tst.fit(X, y['TST'])

model_C.fit(X, y['C'].fillna(y['C'].mean()))
print("save models")
save_catboost_models(model_tst, model_C)
