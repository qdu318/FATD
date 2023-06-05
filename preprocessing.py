import h5py
import numpy as np

filename="data/BJ16_M32x32_T30_InOut.h5"
f = h5py.File(filename)
data=f["data"][:,1,:,:]
data=np.array(data)


np.save("BJ16.npz",data)
