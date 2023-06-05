import h5py
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# 读取数据
filename = "data/BJ14_M32x32_T30_InOut.h5"
f = h5py.File(filename)
data = f["data"][:2000,1,1,1]
data = np.array(data)
# data=data.flatten()

fig = plt.figure(figsize=(12,10))
ax1 = fig.add_subplot(211)
fig = plot_acf(data, lags=15,ax = ax1)
ax2 = fig.add_subplot(212)
fig=plot_pacf(data, lags=15,ax = ax2)

plt.figure(2)
s = pd.Series(data[:100])
pd.plotting.lag_plot(s, lag=1)
plt.show() 