import numpy as np
import fastcluster

def dist(R):
    N = R.shape[0]
    d = R[np.triu_indices(N,1)]
    out = fastcluster.average(d).astype(int)
    
    #Genealogy Set
    dend = {i: (np.array([i]),) for i in range(N)}
    for i,(a,b,_,_) in enumerate(out):
        dend[i+N] = (np.concatenate(dend[a]),np.concatenate(dend[b]))

    [dend.pop(i,None) for i in range(N)]

    return dend.values()


def AvLinkC(Dend,R):
    
    N = R.shape[0]
    Rs = np.zeros((N,N))
    
    for (a,b) in Dend:
        Rs[np.ix_(a,b)] = R[a][:,b].mean()

    Rs = Rs+Rs.T
    np.fill_diagonal(Rs,1)
    return Rs


def SingleHC(X, ddof=0, clip=False):
    R = np.corrcoef(X, ddof=ddof, rowvar=False)
    
    if clip:
        R = np.clip(R, -0.98, 0.98)
        np.fill_diagonal(R,1)
    
    d = dist(1 - R)
    corr_hcal = AvLinkC(d, R)
    std = np.outer( X.std(axis=0), X.std(axis=0))
    return corr_hcal * std


