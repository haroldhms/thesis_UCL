import numpy as np
import pandas as pd

def svds(X,k):
    U, s, V = np.linalg.svd(X,full_matrices=False)
    return U[:,:k], s[:k] , V[:k,:]

def f_adaptive(Xtrain, ytrain, Xtest, k1, k2):
#     Ux, Sx, Vx = svds(Xtrain, k1)
#     Z = Ux
    ZtY = Xtrain.T @ ytrain
    Uzy, Szy, Vzy = svds(ZtY, k2)
    M = Uzy @ np.diag(Szy) @ Vzy
    predY = Xtrain @ M
    predY = predY.T
    predY_test = Xtest.T @ M
    predY_test = predY_test.T
    return predY, predY_test, M


# Ux, Sx, Vx = svds(Xtrain,k1)
# Z = Ux
# ZtY = Z.T @ ytrain
# Uzy, Szy, Vzy = svds(ZtY, k2);
# M = Vx.T @ np.diag(1/Sx) @ Uzy @ np.diag(Szy) @ Vzy
# predY = Xtrain@M
# predY = predY.T
# predY_test = Xtest.T@M
# predY_test = predY_test.T
