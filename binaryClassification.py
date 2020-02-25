import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection

col_names = ['Variance of Wavelet Transformed Image',
            'Skewness of Wavelet Transformed Image',
            'Kurtosis of Wavelet Transformed Image',
            'Entropy of Iimage',
            'Class' ]
# load dataset

data_bank = pd.read_csv("data_banknote_authentication.txt", names=col_names)

print(data_bank)




