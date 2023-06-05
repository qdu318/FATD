import numpy as np
from TSARIMA import FATD
from TSARIMA.util.utility import get_index
import h5py
import numpy as np

if __name__ == "__main__":
    # prepare data
    # the data should be arranged as (ITEM, TIME) pattern
    # import traffic dataset

    filename = "data/BJ16_M32x32_T30_InOut.h5"
    f = h5py.File(filename)
    data = f["data"][:1215, 1, :, :]
    data = np.array(data)
    data=np.moveaxis(data, 0, -1)
    print("shape of data: {}".format(data.shape))
    print("This dataset have {}*{} series, and each serie have {} time step".format(
        data.shape[0],data.shape[1], data.shape[2]
    ))

    # parameters setting
    ts = data[..., :-1] # training data,
    label = data[..., -1] # label, take the last time step as label
    p = 2 # p-order
    d = 1 # d-order
    q = 2 # q-order
    s=48
    P=2
    Q=1
    Rs = [5,5] # tucker decomposition ranks
    k =  10 # iterations
    tol = 0.001 # stop criterion
    Us_mode = 4 # orthogonality mode

    # Run program
    # result's shape: (I
    # TEM, TIME+1) ** only one step forecasting **
    model = FATD(ts, p, d, q,s,P,Q, Rs, k, tol, verbose=1, Us_mode=Us_mode)
    result, _ = model.run()
    pred = result

    # print extracted forecasting result and evaluation indexes
    print("forecast result(first 10 series):\n", pred[:1])
    print("real result(first 10 series):\n", label[:1])
    print("Evaluation index: \n{}".format(get_index(pred, label)))
