from pandas import read_csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np

# current_dir = os.path.dirname(os.path.abspath(__file__))
#
# df = read_csv(f"{current_dir}/data/phpGUrE90.csv")
# v1 = df["V1"].values
# v2 = df["V2"].values
# v8 = df["V8"].values
# shorten_ds = {"v1": v1, "v2": v2, "v8": v8}
# work_df = pd.DataFrame(shorten_ds, columns=["V1", "V2", "V8"])
# print(work_df)
#
# X = df[["V2", "V8"]]
# Y = df["V1"]
#
# regr = linear_model.LinearRegression()
# regr.fit(X, Y)
#
# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)
#
# X = sm.add_constant(X)  # adding a constant
#
# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)
#
# print_model = model.summary()
# print(print_model)
start = 1
X = np.empty((50, 1))
for i in range(len(X)):
    X[i][0] = start
    start += 1
B = np.random.randint(10000, size=(50, 1))
y = np.empty((50, 1))
start = 1
for i in range(len(y)):
    y[i][0] = start * 10 + 15
    start += 1
F = np.empty((50, 2))
for i in range(len(F)):
    F[i][0] = X[i][0]
    F[i][1] = B[i][0]
regr = linear_model.LinearRegression()
regr.fit(F, y)
print(regr.score(F, y))

