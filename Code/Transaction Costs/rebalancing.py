import numpy as np
import pandas as pd

def reb(cca_w,ret, tcost):
    return constant_rebalancing(cca_w, rho=0, gamma=10e-6, kappa=tcost, returns=ret, lag=1, target="Target")
