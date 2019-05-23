import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams

#import tensorflow as tf

import sklearn.preprocessing as skprep
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
import sklearn.metrics as skm
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor



df = pd.read_csv("dmrc_data.csv")

df = df[df["U"] == 4.0]

## This looks like the data I plotted before, which is encouraging
#plt.scatter(df["site_pop"], df["pair_density"])
#plt.show()

y = df["pair_density"]
#X = df[["R1", "R2", "R3", "R4", "L4", "L3", "L2", "L1", "site_pop", "cR1", "cR2", "cR3", "cR4", "cL4", "cL3", "cL2", "cL1"]]
#X = df[["R1", "R2", "R3", "R4", "L4", "L3", "L2", "L1", "site_pop"]]
#X = df[["R1", "R2", "R3", "L4", "L3", "L2", "site_pop"]]
X = df[["R1", "L4", "site_pop"]]
#X = df[["R1", "L4", "site_pop", "cR1", "cL4"]]
#X = df[["site_pop"]]

X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.2, random_state=1)

#poly = skprep.PolynomialFeatures(1)
#X_poly = poly.fit_transform(X)
#X_train, X_test, y_train, y_test = skms.train_test_split(X_poly, y, test_size=0.2, random_state=1)

#regr = sklm.LinearRegression()

## FFNN

layer = (80, 80, 20, 20, 20, 20, 10,10)
regr = MLPRegressor(hidden_layer_sizes=layer, alpha=0.0001, max_iter=10000)

#rbf_kernel = RBF([0.1])
#regr = KernelRidge(kernel=rbf_kernel, gamma=0.1)

regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_fit = regr.predict(X_train)

print("Mean squared error train: %.8f" % skm.mean_squared_error(y_train, y_fit))
print("Mean squared error test: %.8f" % skm.mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.4f' % skm.r2_score(y_test, y_pred))

#plt.scatter(y_fit, y_train, color='blue', label="Training Data")
plt.scatter(y_pred, y_test, color='orange', label="Testing Data")

plt.plot([-1,1],[-1,1], color="black")
plt.ylabel("pair density from FCI")
plt.xlabel("Fit")
#plt.xlim([np.min(y_pred)-0.005,np.max(y_pred)+0.005])
#plt.ylim([np.min(y)-0.005, np.max(y)+0.005])
plt.xlim([0., 0.6])
plt.ylim([0., 0.6])
plt.show()
#plt.savefig("ffnn_s3c0L8.png", bbox_inches='tight', pad_inches=0.05)



