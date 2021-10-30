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


print("cross val")
val = cross_val(X, y, model_tst, model_C, metric)

print(val)
print()
print(np.mean([i[0] for i in val]))
print()
print(np.mean([i[1] for i in val]))

