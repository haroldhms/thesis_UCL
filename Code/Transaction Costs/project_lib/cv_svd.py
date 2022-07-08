
import numpy as np
from sklearn.isotonic import IsotonicRegression as IR


import numpy as np
from sklearn.isotonic import IsotonicRegression as IR


def CV_SVD(X, Y, K=10, df=2, isotonic=True, scaling=True):
    
    # Wrapper
    T, N = X.shape
    T -= df
    CX = np.cov(X, rowvar=False)
    CY = np.cov(Y, rowvar=False)
    C  = Y.T @ X / T - np.mean(Y, axis=0) * np.mean(X, axis=0)
    CC = C @ C.T
    
    
    # SVD of the original data
    U, s, V = np.linalg.svd(C, full_matrices=False)
    
    # generate CV singular values
    scv = cross_validate_svals(X.T, Y.T, K, df)
    
    
    if scaling:
        scale = max([0, (T * CC.trace() - CX.trace() * CY.trace()) 
                       / (T + 1 - 2 * T**(-1))]) / np.linalg.norm(scv) ** 2
#         scale = CC.trace() / np.linalg.norm(scv) ** 2
        scv *= np.sqrt(scale)
    
    if isotonic:
        # run isotonic regression
        scv = IR().fit_transform(s, scv)
    
    return U @ np.diag(scv) @ V, scv


def cross_validate_svals(X, Y, K, df):
    N, T = Y.shape
    scv = np.zeros(N)

    # observations per fold
    obs_fold = T / float(K)
    
    # observations in small and observations in large K
    lowern = np.floor(obs_fold)
    uppern = np.ceil(obs_fold)
    lowern, uppern = int(lowern), int(uppern)

    # how many large and how many small K
    upperf = T - lowern * K
    lowerf = K - upperf
    
    np.random.seed(42)
    permut = np.random.permutation(T)
    
    for i in range(K):
        cv_idx1 = np.zeros([T], dtype=bool)
        
        # get the indices for the singular value estimation
        if i < lowerf:
            cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
        else:
            cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
                           lowerf * lowern + (i - lowerf + 1) * uppern]] = True
        
        cv_idx2 = ~cv_idx1
        
        Gin = Y[:, cv_idx2] @ X[:, cv_idx2].T / (np.sum(cv_idx2)-df) - np.mean(Y[:, cv_idx2], axis=1) * np.mean(X[:, cv_idx2], axis=1)
        Uin, _, Vin = np.linalg.svd(Gin, full_matrices=False)
        Gout = Y[:, cv_idx1] @ X[:, cv_idx1].T / (np.sum(cv_idx1)-df) - np.mean(Y[:, cv_idx1], axis=1) * np.mean(X[:, cv_idx1], axis=1)
        
        scv += np.diag(Uin.T @ Gout @ Vin.T) / K
    
    return np.maximum(scv, 0)


# def CV_SVD(X, Y, K=10, df=2, isotonic=True, scaling=True):
    
#     # Wrapper
#     T, N = X.shape
#     T -= df
#     CX = np.cov(X, rowvar=False)
#     CY = np.cov(Y, rowvar=False)
#     C  = Y.T @ X / T - np.mean(Y, axis=0) * np.mean(X, axis=0)
#     CC = C @ C.T
    
#     # SVD of the original data
#     U, s, V = np.linalg.svd(C, full_matrices=False)
    
#     # generate CV singular values
#     scv, delta = cross_validate_svals(X.T, Y.T, K, df)
    
#     if scaling:
#         scale = max([0, (T * CC.trace() - CX.trace() * CY.trace()) 
#                        / (T + 1 - 2 * T**(-1))]) / np.linalg.norm(scv) ** 2
# #         scale = CC.trace() / np.linalg.norm(scv) ** 2
#         scv *= np.sqrt(scale)
    
#     if isotonic:
#         # run isotonic regression
#         scv = IR().fit_transform(s, scv)
    
    
#     return (1-delta)*U @ np.diag(scv) @ V + delta * np.trace(C) / N *np.eye(N), scv


# def cross_validate_svals(X, Y, K, df):
#     N, T = Y.shape
#     scv = np.zeros(N)

#     # observations per fold
#     obs_fold = T / float(K)
    
#     # observations in small and observations in large K
#     lowern = np.floor(obs_fold)
#     uppern = np.ceil(obs_fold)
#     lowern, uppern = int(lowern), int(uppern)

#     # how many large and how many small K
#     upperf = T - lowern * K
#     lowerf = K - upperf
    
#     np.random.seed(42)
#     permut = np.random.permutation(T)
    
#     for i in range(K):
#         cv_idx1 = np.zeros([T], dtype=bool)
        
#         # get the indices for the singular value estimation
#         if i < lowerf:
#             cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
#         else:
#             cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
#                            lowerf * lowern + (i - lowerf + 1) * uppern]] = True
        
#         cv_idx2 = ~cv_idx1
        
#         Gin = Y[:, cv_idx2] @ X[:, cv_idx2].T / (np.sum(cv_idx2)-df) 
#         Uin, _, Vin = np.linalg.svd(Gin)
#         Gout = Y[:, cv_idx1] @ X[:, cv_idx1].T / (np.sum(cv_idx1)-df) 
#         G = Y @ X.T / T
#         U, _, V = np.linalg.svd(G)
#         target = np.eye(N) 
#         S_target = U @ np.diag(U.T @ target @ V.T) @ V

#         delta = np.trace((target - S_target) @ Gout) / np.linalg.norm(target - S_target, "fro") ** 2 
#         print(delta)
#         scv += 1/(1-delta)*np.diag(Uin.T @ Gout @ Vin.T) / K - delta/(1-delta) * np.diag(Uin.T @ target @ Vin.T) / K
    
#     return np.maximum(scv, 0), delta



# import numpy as np
# from sklearn.isotonic import IsotonicRegression as IR


# def CV_SVD(X, Y, K=10, df=2, isotonic=True, scaling=True):
    
#     # Wrapper
#     T, N = X.shape
#     T -= df
#     CX = np.cov(X, rowvar=False)
#     CY = np.cov(Y, rowvar=False)
#     C  = Y.T @ X / T - np.mean(Y, axis=0) * np.mean(X, axis=0)
#     CC = C @ C.T
    
    
#     # SVD of the original data
#     U, s, V = np.linalg.svd(C, full_matrices=False)
    
#     # generate CV singular values
#     scv = cross_validate_svals(X.T, Y.T, K, df)
    
    
#     if scaling:
#         scale = max([0, (T * CC.trace() - CX.trace() * CY.trace()) 
#                        / (T + 1 - 2 * T**(-1))]) / np.linalg.norm(scv) ** 2
# #         scale = CC.trace() / np.linalg.norm(scv) ** 2
#         scv *= np.sqrt(scale)
    
#     if isotonic:
#         # run isotonic regression
#         scv = IR().fit_transform(s, scv)
    
#     return U @ np.diag(scv) @ V, scv


# def cross_validate_svals(X, Y, K, df):
#     N, T = Y.shape
#     scv = np.zeros(N)

#     # observations per fold
#     obs_fold = T / float(K)
    
#     # observations in small and observations in large K
#     lowern = np.floor(obs_fold)
#     uppern = np.ceil(obs_fold)
#     lowern, uppern = int(lowern), int(uppern)

#     # how many large and how many small K
#     upperf = T - lowern * K
#     lowerf = K - upperf
    
#     np.random.seed(42)
#     permut = np.random.permutation(T)
    
#     for i in range(K):
#         cv_idx1 = np.zeros([T], dtype=bool)
        
#         # get the indices for the singular value estimation
#         if i < lowerf:
#             cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
#         else:
#             cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
#                            lowerf * lowern + (i - lowerf + 1) * uppern]] = True
        
#         cv_idx2 = ~cv_idx1
        
#         Gin = Y[:, cv_idx2] @ X[:, cv_idx2].T / (np.sum(cv_idx2)-df) - np.mean(Y[:, cv_idx2], axis=1) * np.mean(X[:, cv_idx2], axis=1)
#         Uin, _, Vin = np.linalg.svd(Gin, full_matrices=False)
#         Gout = Y[:, cv_idx1] @ X[:, cv_idx1].T / (np.sum(cv_idx1)-df) - np.mean(Y[:, cv_idx1], axis=1) * np.mean(X[:, cv_idx1], axis=1)
        
#         scv += np.diag(Uin.T @ Gout @ Vin.T) / K
    
#     return np.maximum(scv, 0)


# import numpy as np

# def CV_SVD(X, Y, K=10, df=2):
    
#     # Wrapper
#     T, N = X.shape
#     T -= df
#     S  = Y.T @ X / T - np.mean(Y, axis=0) * np.mean(X, axis=0)
    
#     # generate CV singular values
#     delta = cross_validate_svals(X.T, Y.T, K, df)
    
#     return delta * S + (1 - delta) * np.eye(N) 


# def cross_validate_svals(X, Y, K, df):
#     N, T = Y.shape
#     delta = 0

#     # observations per fold
#     obs_fold = T / float(K)
    
#     # observations in small and observations in large K
#     lowern = np.floor(obs_fold)
#     uppern = np.ceil(obs_fold)
#     lowern, uppern = int(lowern), int(uppern)

#     # how many large and how many small K
#     upperf = T - lowern * K
#     lowerf = K - upperf
    
#     np.random.seed(42)
#     permut = np.random.permutation(T)
    
#     for i in range(K):
#         cv_idx1 = np.zeros([T], dtype=bool)
        
#         # get the indices for the singular value estimation
#         if i < lowerf:
#             cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
#         else:
#             cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
#                            lowerf * lowern + (i - lowerf + 1) * uppern]] = True
        
#         cv_idx2 = ~cv_idx1
        
#         Gin = Y[:, cv_idx2] @ X[:, cv_idx2].T / (np.sum(cv_idx2)-df) 
#         Uin, sin, Vin = np.linalg.svd(Gin)
#         Gin_sub = Uin[:,:3] @ np.diag(sin[:3]) @ Vin[:3,:]
#         Gout = Y[:, cv_idx1] @ X[:, cv_idx1].T / (np.sum(cv_idx1)-df) 
#         target = np.eye(N) / N * np.trace(Gin_sub)
        
#         G = Y @ X.T / T
#         delta += np.linalg.norm(G - Gout, "fro") ** 2 / ( np.linalg.norm(G - Gout, "fro") ** 2 + np.linalg.norm(target - Gout, "fro") ** 2 )
#         print(delta)
#         delta /= K
    
#     return delta


# class SVDCleaning(object):
#     def __init__(self, Y, X, method='id'):
#         self.Y = Y
#         self.X = X
#         self.Yhat = None
#         self.Xhat = None
#         self.T = None
#         self.N = None
#         self.method = 'id'
#         self.cov_x = np.cov(self.X, rowvar=False)
#         self.cov_y = np.cov(self.Y, rowvar=False)
#         self.vol = None

        
#     def decorrelate(self, X, D, V):
#         """Decorrelate the variables to have unit variance

#         :param X: return or signal matrix
#         :return: decorrelated matrix, eigevalues, eigenvectors
#         """
#         Xhat = X @ V @ np.diag(1/ np.sqrt(D)) / np.sqrt(self.T)
#         return Xhat
    
    
#     def construct_cov(self, D, V):
#         """Reconstruct covariance matrix from eigenvalues and inverted eigenvectors of covariance matrix

#         :param D: Eigenvalues of covariance matrix
#         :param V: Eigenvectors of covariance matrix
#         :return: Reconstructed covariance matrix
#         """
#         v_inv = np.linalg.inv(V)
#         d_mat = np.diag(D)
#         return v_inv.dot(d_mat.dot(V))
    
    
#     def estimate_eig(self, method='id'):
#         """Obtain estimated eigenvalues

#         :param Y: Returns array
#         :param X: Signals array
#         :param method: Method to estimate cross-covariance (id or svd)
#         :return: Reconstructed covariance matrix
#         """
#         # extract sample eigenvalues and eigenvectors
#         self.T, self.N = self.Y.shape
        
#         T, N = data.shape
    
#         assert data.shape[0] > data.shape[1]


#         cov = np.cov(data, rowvar=False, ddof=df)
#         d, v = np.linalg.eigh(cov)

#         if self.method == 'sample':
#             pass

#         elif self.method == 'nls':
#             _, d = NLSHRINK(data.T, df=df)

#         elif method == 'cv':
#             _, d = CV(data, df=df)

#         elif self.method == 'ls':
#             LW = LedoitWolf().fit(ret)
#             coef = LW.shrinkage_
#             d = (1-coef) * d + (coef) * np.mean(d)

#         elif self.method == 'id':
#             d = np.ones(N)

#         else:
#             raise NotImplementedError("Return covariance {} not implemented".format(method))

#         return d
    
    
#     def estimate_cross_cov(self):
#         """Obtain estimated covariance

#         :param Y: Returns array
#         :param X: Signals array
#         :param vol: Optional volatility correction if vol-scaled returns is used
#         :param method: Method to estimate cross-covariance (id or svd)
#         :return: Reconstructed covariance matrix
#         """
#         # extract sample eigenvalues and eigenvectors
#         self.T, self.N = self.Y.shape
# #         self.cov = (self.X.T @ self.X) / self.T
#         Dx, Vx = np.linalg.eigh(self.cov_x)
#         Dy, Vy = np.linalg.eigh(self.cov_y)
    
#         # compute decorrelate variables
#         self.Xhat = self.decorrelate(self.X, Dx, Vx)
#         self.Yhat = self.decorrelate(self.Y, Dy, Vy)
        
#         # compute sample cross-corr
#         G = self.Yhat.T @ self.Xhat
        
#         # estimate cross-corr
#         # est_cross_corr, _ = CV(Xhat, Yhat)
#         est_cross_corr, _ = RIE_Cross_Correlation(G, self.T)
        
#         cov_half_y = self.construct_cov(np.sqrt(Dy), Vy)
#         cov_half_x = self.construct_cov(np.sqrt(Dx), Vx)
        
#         cross_cov = cov_half_y @ est_cross_corr @ cov_half_x
            
#         return est_cross_cov



# Estimate pi
# Xm = X - X.mean(axis=0)
# y = Xm ** 2
# pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
# pi_hat = np.sum(pi_mat)


# def func(X, Y):
#     T, N = Y.shape
# #     X -= X.mean(axis=0)
# #     Y -= Y.mean(axis=0)
#     G = Y.T @ X / T
#     U, _, V = np.linalg.svd(G)
# #     _, s_rie = RIE_Cross_Correlation(G, T)
# #     Syx = U @ np.diag(s_rie) @ V
#     U, s, V = np.linalg.svd(G)
#     Syx = U[:,:3] @ np.diag(s[:3]) @ V[:3,:]
    
#     # Shrinkage target
#     mu = np.trace(Syx) / N
#     F = mu * np.eye(N)
    
#         # Shrinkage target
# #     U, s, V = np.linalg.svd(corr_yx)
# #     corr_yx = U @ np.diag(s_rie) @ V
 
#     X2 = X ** 2
#     Y2 = Y ** 2
#     G2 = np.dot(Y2.T, X2) / T
#     U, s_rie2, V = np.linalg.svd(G2)
# #     _, s_rie2 = RIE_Cross_Correlation(G2, T)
#     Syx2 = G2
#     U2, s2, V2 = np.linalg.svd(G2)
#     Syx2 = U2[:,:3] @ np.diag(s2[:3]) @ V2[:3,:]
#     pi_mat = Syx2 - Syx ** 2
    
#     phi = np.sum(pi_mat)
#     gamma_hat = np.linalg.norm(Syx - F, "fro") ** 2
#     kappa = phi / gamma_hat
#     shrinkage = np.maximum(0, np.minimum(1, kappa/T))
#     return shrinkage