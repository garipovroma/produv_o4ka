import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import *
from utils.model import *

print("read test data")
test_data = read_data('test')
print("preprocess data")
X_test = preprocess_data(test_data)
print("get catboost models")
model_tst, model_C = get_catboost_models()
print("predict data")
y = predict(X_test, model_tst, model_C)
print("saving to csv")
save_to_csv(y)
