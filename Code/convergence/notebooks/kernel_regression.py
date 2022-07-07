from localreg import *
import pandas as pd
import numpy as np
def kernel_regression(X,R,kernel,col):
    print("col : {}".format(col))
    a = localreg(np.array(X[:,col]), np.array(R[:,col]),
         x0=None, degree=0, kernel=kernel, radius=1, frac=None)
    return a
