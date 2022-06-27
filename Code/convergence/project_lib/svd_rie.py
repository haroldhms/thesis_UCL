import numpy as np
from sklearn.isotonic import IsotonicRegression as IR


def RIE_Cross_Covariance(Etotale, T, n, p, Return_Ancient_SV=False, Return_New_SV=False, Return_Sing_Vectors=False, adjust=False, exponent_eta=.5, C_eta=1):
    """Flo's algo. We need n\leq p, Etotale needs to be of the type np.matrix"""    
    U, s, V = np.linalg.svd(Etotale[np.ix_(range(n),range(n,n+p))],full_matrices=1)
    Coeff_A, Coeff_B, Coeff_B_n_to_p = [], [], 0
    
    for k in range(n):
        Coeff_A.append((U[:,k].T*(Etotale[np.ix_(range(n),range(n))])*U[:,k])[0,0])
        Coeff_B.append((V[k,:]*(Etotale[np.ix_(range(n,n+p),range(n,n+p))])*(V[k,:].T))[0,0])
        
    for k in range(n,p):
        Coeff_B_n_to_p+=(V[k,:]*(Etotale[np.ix_(range(n,n+p),range(n,n+p))])*(V[k,:].T))[0,0]
        
    stwo = s**2    
    eta = C_eta*(n*p*T)**(-exponent_eta/3.)
    new_s = []
    
    for k in range(n):
#        z=s[k]+1j*eta
        ztwo = (s[k]+1j*eta)**2
        one_over_ztwo_minus_stwo = 1/(ztwo-stwo)
        TH = np.dot(stwo,one_over_ztwo_minus_stwo)
#        TA=np.dot(Coeff_A,one_over_ztwo_minus_stwo)
#        TB=np.dot(Coeff_B,one_over_ztwo_minus_stwo)+Coeff_B_n_to_p/ztwo
#        TTheta=ztwo*TA*TB/(T+TH)
        new_s.append(s[k]*np.max([-np.imag(T**2/(T+TH-(ztwo*np.dot(Coeff_A,one_over_ztwo_minus_stwo)*(np.dot(Coeff_B,one_over_ztwo_minus_stwo)+Coeff_B_n_to_p/ztwo)/(T+TH))))/np.imag(TH),0]))
#        new_s.append(np.max([-np.imag(T**2/(T+TH-(ztwo*np.dot(Coeff_A,one_over_ztwo_minus_stwo)*(np.dot(Coeff_B,one_over_ztwo_minus_stwo)+Coeff_B_n_to_p/ztwo)/(T+TH))))/np.imag(z*np.sum(one_over_ztwo_minus_stwo)),0]))
    if adjust:
        new_s = np.array(new_s)
        new_s *= np.sqrt(max([0, (T * sum(stwo) - np.trace(Etotale[np.ix_(range(n), range(n))]) * np.trace(Etotale[np.ix_(range(n, n + p),range(n, n + p))])) / (T + 1 - 2 * T**(-1))])) / np.linalg.norm(new_s)
        # new_s*=np.linalg.norm(s)/np.linalg.norm(new_s)
    U, V = np.matrix(U), np.matrix(V[range(n),:]).T
    
    new_s = IR().fit_transform(s, new_s)
    L = [U*np.diag(new_s)*(V.T)]
    
    if Return_Ancient_SV:
        L.append(s)
    if Return_New_SV:
        L.append(new_s)
    if Return_Sing_Vectors:
        L.append(U)
        L.append(V)
    if Return_Ancient_SV or Return_New_SV or Return_Sing_Vectors:
        return tuple(L)
    else:
        return L[0]
       
def RIE_Cross_Correlation(Ee, T, Return_Ancient_SV=False, Return_New_SV=False):
    """Flo's algo. We need n\leq p. The previous algo works a bit better (but a bit slower)"""
    n, p = Ee.shape
    alpha, beta = n/float(T), p/float(T)
    U, s, V = np.linalg.svd(Ee,full_matrices=False)
    eta = (n*p*T)**(-1/6.)
    new_s = []
    stwo = s**2
    for k in range(n):
        ztwo = (s[k]+1j*eta)**2
        G = np.sum(1/(ztwo-stwo))/float(T)
        #wtG=G+(beta-alpha)/ztwo
        H = ztwo*G-alpha
        #K = z**2*G*wtG*(1+H)**2
        #approx_L=(1+2*H-np.sqrt(1+4*K))/(2+2*H)
        new_s.append(s[k]*np.max([np.imag((1+2*H-np.sqrt(1+4*ztwo*G*(G+(beta-alpha)/ztwo)*(1+H)**2))/(2+2*H))/(np.imag(H)),0]))
#        new_s.append(np.max([np.imag((1+2*H-np.sqrt(1+4*ztwo*G*(G+(beta-alpha)/ztwo)*(1+H)**2))/(2+2*H))/(np.imag(z*G)),0]))
    
    new_s = IR().fit_transform(s, new_s)
    
    if (Return_Ancient_SV and Return_New_SV):
        return (np.matrix(U)*np.diag(new_s)*V,s,new_s)
    elif Return_Ancient_SV:
        return (np.matrix(U)*np.diag(new_s)*V,s)
    elif Return_New_SV:
        return (np.matrix(U)*np.diag(new_s)*V,new_s)
    else:
        return np.array(np.matrix(U)*np.diag(new_s)*V), new_s
    
    
# class SVDCleaning(object):
#     def __init__(self, Y, X, vol=None):
#         self.Y = Y
#         self.X = X
#         self.Yhat = None
#         self.Xhat = None
#         self.T = None
#         self.N = None
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
        
#         est_cov_half_y = construct_cov(np.sqrt(Dy), Vy)
#         est_cov_half_x = construct_cov(np.sqrt(Dx), Vx)
        
#         est_cross_cov = est_cov_half_y @ est_cross_corr @ est_cov_half_x
        
#         if self.vol is not None:
#             est_cross_cov = corr2cov(est_cross_cov, self.vol)
            
#         return est_cross_cov