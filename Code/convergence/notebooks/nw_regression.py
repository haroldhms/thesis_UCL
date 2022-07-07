from statsmodels.nonparametric.kernel_regression import KernelReg
import pandas as pd
import numpy as np
def nw_regression(X,R,col):
    print("col : {}".format(col))
    kr = KernelReg(np.array(R[:,col]),np.array(X[:,col]),var_type='c')
    estimator = kr.fit()
    return estimator[0]
