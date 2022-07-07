

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn.metrics.pairwise import rbf_kernel
from statsmodels.nonparametric.kernel_regression import KernelReg
from localreg import *
import multiprocessing as mp
import time

############### kernel functions ##############


def nw_regression(X, R, col):
    """
        Nadaraya-Watson regression
        inputs: X : matrix of signals [m x l]
                R : matrix of returns [m x l]
                col : column to perform regression on [1 x 1]
        outputs:
                Univariate Nadaraya-Watson regression of return on signal [1 x l]
    """
    kr = KernelReg(np.array(R[:, col]), np.array(X[:, col]), var_type='c')
    estimator = kr.fit()
    return estimator[0]


def gaussian_kernel(X, s=1):
    """
    function to obtain gaussian kernel (K(x, y) = exp(-gamma ||x-y||^2))
    inputs: X : numpy matrix of size [l x m]
            s : scale factor         [1 x 1]
    output : gaussian kernel [l x l]
    """
    # manually
    #pairwise_dists = squareform(pdist(X, 'euclidean'))
    #K = scipy.exp(-pairwise_dists ** 2 / s ** 2)

    # built-in
    K = rbf_kernel(X, gamma=s)
    return K

# used in nonparametric cca, 2016, Livescu, Michaeli & Wang


def kernel_regression(X, R, kernel, col):
    """
        function to calculate univariate kernel regression of asset on signal
        inputs:
            X  : matrix of signals [m x l]
            R  : matrix of returns [m x l]
            kernel : kernel type [string] -> rbf.triangular, rbf.gaussian, rbf.epanechnikov
            col : column for which to calculate regression [1 x 1]
        output:
            numpy array of kernel regression result [1 x l]

    """
    output = localreg(np.array(X[:, col]), np.array(R[:, col]),
                      x0=None, degree=0, kernel=kernel, radius=1, frac=None)

    return output

# used in nonparametric cca, 2016, Livescu, Michaeli & Wang


def parallel_kernel(X, R, kernel):
    """
        function to calculate kernel regressions of each asset
        inputs:
            X : matrix of signals [m x l]
            R : matrix of returns [m x l]
            kernel : kernel type [string] -> rbf.triangular, rbf.gaussian, rbf.epanechnikov
        output:
            numpy array of univariate kernel regressions [m x l]
    """
    # initialize pool
    pool = mp.Pool(mp.cpu_count())
    # parallelized kernel regression of signals
    results = pool.starmap(kernel_regression, [(
        X, R, kernel, a) for a in np.arange(0, R.shape[1], 1)])
    results = np.array(results).T
    pool.close()
    pool.join()
    return results


############### partially linear CCA using CVX conversion ##########
# from Shawe-Taylor, 2008
def cca_deflate(trainX, Kb, k, sk, verbose, solver):
    """

     This is a wrapper function that runs the SCCA2 function while deflating
     and finding new projection directions

     Input:
       trainX : returns matrix [m x l]
       Kb     : signals kernel matrix [l x l]
       k      : vector of indices for deflations [1 x n]
       sk     : scaling parameter for mu and gamma [1 x 1]
       verbose: whether you want verbose output [boolean]
       solver : what solver to use [string] -> ECOS, ECOS_BB, SCS
     Output:
       output : a tuple struct with various values

    """
    tX = trainX.copy()
    KK = Kb.copy()
    if type(KK) != np.ndarray:
        KK = np.array(KK)
    co = 0

    # initialise temporary variables
    wa = np.zeros((trainX.shape[0], len(k)))
    e = np.zeros((Kb.shape[1], len(k)))
    corval = np.zeros(len(k))
    resid = np.zeros(len(k))
    projk = np.zeros((Kb.shape[1], len(k)))
    tau = np.zeros((Kb.shape[1], len(k)))
    proj = np.zeros((trainX.shape[0], len(k)))
    t = np.zeros((Kb.shape[1], len(k)))

    for i in range(len(k)):
        # print('.')
        output_w, output_e, output_cor, output_res = plcca(
            trainX, KK, k[i], sk, verbose, solver)
        wa[:, co] = output_w
        e[:, co] = output_e
        corval[co] = output_cor
        resid[co] = output_res

        co += 1

        # dual deflation
        projk[:, i] = KK @ e[:, i]
        tau[:, i] = KK @ projk[:, i]

        P = np.eye(len(KK)) - \
            (np.outer(tau[:, i], tau[:, i])) / (tau[:, i].T @ tau[:, i])
        KK = P.T @ KK @ P
        # primal deflation
        proj[:, i] = tX @ (tX.T @ wa[:, i])
        t[:, i] = tX.T @ proj[:, i]
        tX = tX - tX @ (np.outer(t[:, i], t[:, i])) / (t[:, i].T @ t[:, i])
    #print('    ')

    # Primal projection
    P = trainX @ t @ np.linalg.inv(t.T @ t)
    W = proj @ np.linalg.inv(P.T @ proj)

    # can't think of a fancy way to normalise the vectors
    WW = W.copy()
    for i in range(W.shape[1]):
        WW[:, i] = W[:, i] / np.linalg.norm(trainX.T @ W[:, i])
    W = WW

    # Dual Projection
    Z = projk @ np.linalg.inv(np.linalg.inv(tau.T @ tau) @ tau.T @ Kb @ projk)

    ZZ = Z.copy()
    for i in range(Z.shape[1]):
        ZZ[:, i] = Z[:, i] / np.linalg.norm(Kb @ Z[:, i])
    Z = ZZ

    output = {"w": wa,
              "e": e,
              "P": P,
              "dual_tau": tau,
              "primal_tau": t,
              "cor": corval,
              "res": resid,
              "W": W,
              "Z": Z,
              }
    return output


def plcca(X, K, masterI, sk, verbose=False, solver="ECOS"):
    """
    based on https://github.com/aalto-ics-kepaco/primal_dual_scca

     Original description by David R. Hardoon: 
     Sparse Canonical Correlation Analysis - SCCA, is a primal-dual solver for
     the CCA problem. Given primal data of a view and a dual representation of
     the second view will provide a sparse primal weight vector (for the primal
     data) and sparse feature projection (for the dual [kernel] data)

     Input:  X             : Primal data of view one    [m x l] (rows is the number of assets)
             K             : dual data of view two      [l x l]
             seed_index    : Starting point for e       [1 x 1]
             sk            : scaling factor for mu and gamma

     Output: w             : sparse weight vector      [1 x m]
             e             : sparse projct vectors     [1 x l]
             cor           : correlation value         [1 x 1]
             res           : residual value            [1 x 1]
    """

    # Initialising parameters
    e = np.zeros(K.shape[1])
    e[masterI] = 1
    c = X @ (K[:, masterI] * e[masterI])
    KK = K.T @ K
    # More setting up initial parameters
    M = X.shape[0]
    L = X.shape[1]
    Ij = np.eye(K.shape[1])
    Ij[masterI, masterI] = 0

    # Set trade-off to half
    tau = 0.5

    # Setting the mu and gamma regularsation parameters
    d1 = 2*tau*(1-tau)*c
    mu = sk*np.mean(abs(d1))
    gamma = np.mean(abs(2 * (1-tau)**2 * Ij @ KK[:, masterI]*e[masterI]))  # OK

    # CVX optimisation

    # initialise variables
    w = cp.Variable(M)
    e = cp.Variable(L)

    # example to print whether it is valid constraint :
    # print("constraint is dcp:", (cp.norm_inf(e) >= 0).is_dcp())

    # determine objective function and constraints
    # NOTE : NO OPTION TO CONSTRAIN DUAL VALUE OF e
    obj = cp.Minimize(cp.norm(tau * X.T @ w - (1-tau)*K @ e,
                      2) + mu*cp.norm(w, 2) + gamma*cp.norm(e, 2))
    constraints = [cp.norm_inf(e) <= 1, e[masterI] == 1]

    # solve problem
    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose, warm_start=True)

    # extract correlation
    w = np.array(w.value)
    e = np.array(e.value)
    p1 = w @ X @ X.T @ w
    p2 = e @ K @ K @ e
    corr = w @ X @ K @ e / np.sqrt(p1*p2)

    # return primal weights, dual weights, correlation and residual
    return w, e, corr, prob.value

############### partially linear CCA using exact conversion ##########
# from Shawe-Taylor, 2008


def scca(X, K, masterI, outdis, debug, sk, oo, ww, ee, convergence_analysis=False):
    """
 Sparse Canonical Correlation Analysis - SCCA, is a primal-dual solver for
 the CCA problem. Given primal data of a view and a dual representation of
 the second view will provide a sparse primal weight vector (for the primal
 data) and sparse feature projection (for the dual [kernel] data)

 Input:  X        : Primal data of view one    [m x l]
         K        : dual data of view two      [l x l]
         masterI  : Starting point for e       [1 x 1]
         od       : output display true/false  [boolean]
         debug    : outputs primal-dual progression true/false     [boolean]
         sk       : scaling factor for mu and gamma                [1 x 1]
         oo       : max iterations over outer while loop           [1 x 1]
         ww       : max iterations over primal weights update loop [1 x 1]
         ee       : max iterations over dual weights update loop   [1 x 1]

 Output: w     : sparse weight vector      [1 x m]
         e     : sparse projct vectors     [1 x l]
         alpha : 1'st view dual parameters [1 x m]
         beta  : 2'nd view dual parameters [1 x l]
         mu    : regularsation parameter   [1 x 1]
         gamma : lagrangian (scale factor) [1 x 1]
         cor   : correlation value         [1 x 1]
         res   : Optimisation solution/s   [1 x 1]

         time_o : time (s) of outer while                            [1 x 1]
         time_w : total (s) taken by primal weight convergences      [1 x 1]
         time_e : total (s) taken by dual weight convergences        [1 x 1]
         loops_o : number of loops overall while                     [1 x 1]
         loops_w : total number of loops primal weights while        [1 x 1]
         loops_e : total number of loops dual weights while          [1 x 1]

Based on MATLAB code from David R. Hardoon 25/06/2007
    """
    # initialise debug variables
    time_o = 0
    time_w = 0
    time_e = 0
    loops_o = 0
    loops_w = 0
    loops_e = 0

    # Initialising parameters
    # Setting the infinity norm
    e = np.zeros(K.shape[1])
    beta = e.copy()
    e[masterI] = 1
    sa_e = e.copy()
    eDifference = 0

    # So we don't need to recomput do once.
    c = X @ (K[:, masterI] * e[masterI])
    KK = K @ K

    # More setting up initial parameters
    N = X.shape[0]
    w = np.zeros(N)
    sa_w = w.copy()
    j = np.ones(N)

    # So that we do not use the e_i
    Ij = np.eye(K.shape[1])
    Ij[masterI, masterI] = 0

    # Setting initial tolerance values
    etolerance = 0.01
    globtolerance = 1e-5

    # Set trade-off to half
    tau = 0.5

    # Setting the mu and gamma regularsation parameters
    d1 = 2*tau*(1-tau)*c  # The reminder of the equation is zero
    mu = sk*np.mean(abs(d1))

    gamma = np.mean(abs(2 * (1-tau)**2 * Ij @ KK[:, masterI]*e[masterI]))
    # Computing the upper bound on the w's
    C = 2*mu
    # Computing inital alpha
    alpha = d1 + mu*j

    # Finding alphas that break the constraints
    I1 = np.where(alpha < 0)[0]
    I2 = np.where(alpha > C)[0]
    I = np.sort(np.concatenate([I1, I2]))

    # Selecting the violations
    ta = alpha[I]
    ta[ta > 0] = ta[ta > 0] - C
    ta = abs(ta)

    stai = np.argsort(ta)
    I = I[stai]

    if len(I) > 1000:
        I = I[(len(I)-999):]

    pI = I

    # Initial W tolerance is set
    tolerance = 0.3*abs(max(alpha[I]))

    if outdis == 1:
        print("Selected regularisation value; mu = {}, gamma = {}".format(mu, gamma))

    # We don't need to work on all of e
    J = np.where(e != 0)

    # Remembering the alpha violations
    preAlpha = alpha[I]

    # Initially the difference will be zero
    alphaDifference = abs(alpha[I] - preAlpha)

    # Flag on whether to exit
    exitLoop = 1

    # Loop counter
    wloop = 1

    # Do we need to compute the covariance?
    skipCon = 0

    # Do we need to go over all of e, to find new violations
    completeE = 1
    loo = 1

    # The loop is repeated while there are violations and we are still working
    # on the alphas and that the difference between the pervious and corrent
    # alphas is minimal
    # initiate time of overall while loop
    time_o = time.time()
    while((I.any() and exitLoop) | (sum((alphaDifference > globtolerance) == 1) == 0)):
        # set change to true so we enter the convergence on w
        if wloop > oo:
            break
        change = True
        N = len(I)
        # compute the new covariance matrix if needed
        if (skipCon == 0):
            CX = X[I, :] @ X[I, :].T

        # save the previous alphas
        preAlpha = alpha[I]
        # until convergence do
        # set limit on number of converges
        converger_count = 0

        timer_start = time.time()
        while(change):
            converger_count += 1
            if converger_count > ww:
                break
            # we can exit
            change = False
            # setting the update
            lefts = CX @ w[I]

            # for the found alphas
            for i in range(N):
                # upper and lower bounding alpha
                needtoupdate1 = False

                if(alpha[I[i]] > C):
                    alpha[I[i]] = C
                    needtoupdate1 = True
                elif (alpha[I[i]] < 0):
                    alpha[I[i]] = 0
                    needtoupdate1 = True
                else:
                    # if alpha is between the bound values
                    # shift w if needed
                    if (w[I[i]] > 0):
                        dw = (C-alpha[I[i]]) / (2 * tau**2 * CX[i, i])
                        w[I[i]] = w[I[i]] - dw
                    elif (w[I[i]] < 0):
                        dw = alpha[I[i]] / (2 * tau**2 * CX[i, i])
                        w[I[i]] = w[I[i]] + dw

                # update w if needed to
                if (needtoupdate1 == True):
                    # computing the learning rate
                    learningRate = 1 / (2 * tau**2 * CX[i, i])

                    # updating
                    firstBit = 2 * tau * (1-tau) * c[I[i]] + mu - alpha[I[i]]
                    w[I[i]] = w[I[i]] + learningRate * \
                        (firstBit - 2 * tau**2 * lefts[i])

                # Checking that w does not skip zero
                if ((sa_w[I[i]] < 0 and w[I[i]] > 0) or (sa_w[I[i]] > 0 and w[I[i]] < 0)):
                    w[I[i]] = 0

                # computing change
                b = w[I[i]]-sa_w[I[i]]
                sa_w[I[i]] = w[I[i]]

                if b != 0:
                    lefts = lefts + CX[:, i] * b

                # computing the new lagrangian
                alpha[I] = 2 * tau * (1-tau) * c[I] + mu - 2 * tau**2 * lefts

                # did we converge enough?
                if abs(b) > tolerance:
                    change = True

            # for loop ident
        # update debug varibales
        time_w += time.time() - timer_start
        loops_w = converger_count

        ########################################
        # working on the e's now

        # check whether we need to even waste time on e

        if K.shape[1] > 1:
            # compute all beta's (since beta are taking into account as a shadow
            # variable i.e. they are not really computed, we are able to use their
            # value as an indication of which e's are needed)
            local_beta = 2 * (1-tau)**2 * Ij @ (KK @ e) - 2 * \
                tau * (1-tau) * Ij @ K.T  @ X[I, :].T  @ w[I] + gamma
            # find e's that need to be worked on
            J = np.sort(np.append(np.where(local_beta < 0), masterI))

            # save previous e's
            preE = e[J]
            # precompute part of lagrangian update
            # Ij subsetting
            Ij1 = Ij[:, J]
            Ij2 = Ij1[J, :]

            oneP = 2 * tau * (1-tau) * Ij2.T  @ K[:, J].T @ X[I, :].T @ w[I]
            # converging over e
            change = True
            N = J.shape[0]

            # until convergence
            # set a counter

            converger = 0
            time_starter = time.time()
            while(change == True):
                converger += 1
                if converger > ee:
                    break
                change = False
                KK1 = KK[:, J]
                KK2 = KK1[J, :]

                lefts = Ij2.T @ KK2 @ e[J]

                for i in range(N):
                    if (J[i] != masterI):

                        learningRate = 1 / (4 * (1-tau)**2 * KK[J[i], J[i]])
                        #print("learning rate : {}".format(learningRate))

                        if (learningRate > 1e+3 or learningRate < 1e-3):
                            learningRate = 1
                        # before : oneP[i]

                        e[J[i]] = e[J[i]] + learningRate * \
                            (oneP[i] - 2 * (1-tau)**2 *
                             lefts[i] + beta[J[i]] - gamma)

                        if (e[J[i]] < 0):
                            e[J[i]] = 0

                        elif (e[J[i]] > 1):
                            e[J[i]] = 1
                        else:
                            beta[J[i]] = 0
                        b = e[J[i]]-sa_e[J[i]]
                        sa_e[J[i]] = e[J[i]]
                        if b != 0:

                            lefts = lefts + Ij2.T @ KK[J, J[i]] * b

                            if abs(b) > etolerance:
                                change = True
            # update debug variables
            time_e += time.time() - time_starter
            loops_e = converger

            # recompute c
            c = X @ (K[:, J] @ e[J])

            # check to see if there is any difference from previous e's
            eDifference = abs(e[J] - preE)

            # compute new tolerance values
            etolerance = 0.3 * abs(max(eDifference))

            # bound the tolerance values
            if (etolerance == 0 or etolerance < globtolerance):
                etolerance = globtolerance

        # recompute alpha using the new w's
        alpha = 2 * tau * (1-tau) * c + mu*j - 2 * \
            tau**2 * X @ (X[I, :].T @ w[I])

        # check to see if there is any difference from previous alpha's (e's)
        alphaDifference = abs(alpha[I] - preAlpha)
        #print("alpha diff : {}".format(alphaDifference))

        # compute new tolerance values
        tolerance = 0.3*abs(max(alphaDifference))

        if (tolerance == 0 or tolerance < globtolerance):
            tolerance = globtolerance

        if debug:
            print('Loop number {}'.format(wloop))
            print('Tolerance value = {}'.format(tolerance))
            print('Error value = {}'.format(sum(alphaDifference)))
            print('Etolerance value = {}'.format(etolerance))
            print('Error evalue = {}'.format(sum(eDifference)))

        # find alphas that break the constraint
        markI = I.copy()
        skipCon = 0

        I1 = np.where(alpha + globtolerance < 0)[0]
        I2 = np.where(alpha - globtolerance > C)[0]
        I = np.sort(np.concatenate([I1, I2]))

        # breakout if need to
        if (I.any() == True):
            exitLoop = 1
            # selecting the maximum nf violations
            ta = alpha[I]
            ta[ta > 0] = ta[ta > 0] - C
            ta = abs(ta)

            # sorting as to select the largest violations first
            stai = np.argsort(ta)
            I = I[stai]

            # sanity check - are any of the violations repeats?
            for kp in range(len(I)):

                lc = np.where(I[kp] == pI)[0]

                if lc > -10:
                    pI[lc] = -10

            # grab only one copy of the violations
            pI = pI[[pI != -10]]

            # adding the previous I's for which w has a non zero element
            np.sort(np.append(np.where(local_beta < 0), masterI))
            I = np.sort(np.append(pI[w[pI] != 0], I))
            if len(I) > 1000:
                I = I[(len(I) - 999):]
            # check to see if we need to compute the covariance matrix again
            tmp1 = sum(np.tile(markI.T, (len(I), 1)) ==
                       np.tile(I, (len(markI), 1)).T)

            if (sum(tmp1) == len(tmp1) and len(I) == len(markI)):
                skipCon = 1

            # saving the current index
            pI = I

        else:
            # no violations, we can potentially exit the algorithm
            exitLoop = 0
            I = pI

        # update loop number
        loops_o = wloop
        wloop += 1

    ######################################
    # end of convergence algorithm
    # update time of overall while loop
    time_o = time.time() - time_o

    # compute vector length
    wv = (w @ X) @ (X.T @ w)
    ev = e @ KK @ e

    # normalize e
    e = e / np.sqrt(ev)

    # normalize w, but check that we found something
    if sum(w != 0) > 0:
        w = w/np.sqrt(wv)

    # compute the optimisation error value
    res = np.linalg.norm(tau * X.T @ w - (1-tau) * K @ e)**2

    # compute the correlation value
    cor = w @ X @ K @ e

    if outdis == 1:
        print('----------------------------------------------- \n')
        print('we have {} non zero weights'.format(sum(w != 0)))
        print('and {} non zero dual weights'.format(sum(e != 0)))
        print('correlation = {}'.format(cor))
        print('mu = {}'.format(mu))
        print('gamma = {}'.format(gamma))
        tmp = e[masterI]
        e[masterI] = 0
        print('|e|1 = {}'.format(np.linalg.norm(e, 1)))
        e[masterI] = tmp
        print('e*KK*e = {}, w*X*X*w = {}'.format(ev, wv))
        print('----------------------------------------------------')

    # return times & number of loops if performing convergence analysis
    if convergence_analysis == True:
        return w, e, alpha, beta, mu, gamma, cor, res, loops_o, loops_w, loops_e, time_o, time_w, time_e

    else:
        return w, e, alpha, beta, mu, gamma, cor, res


def scca_deflator(trainX, Kb, k, outdis, debug, sk, oo, ww, ee):
    """

     This is a wrapper function that runs the SCCA2 function while deflating
     and finding new projection directions

     Input:
       trainX : first view train data
       Kb     : second view train data
       k      : vector of indices for deflations
       outdis, debug    : output and debug variables for scca
        sk    : scaling factor for mu and gamma
     Output:
       output : a struct with various values

     based on MATLAB code by David R. Hardoon
    """
    tX = trainX.copy()
    KK = Kb.copy()
    co = 0

    # initialise temporary variables
    wa = np.zeros((trainX.shape[0], len(k)))
    e = np.zeros((Kb.shape[1], len(k)))
    resval = np.zeros(len(k))
    corval = np.zeros(len(k))
    projk = np.zeros((Kb.shape[1], len(k)))
    tau = np.zeros((Kb.shape[1], len(k)))
    proj = np.zeros((trainX.shape[0], len(k)))
    t = np.zeros((Kb.shape[1], len(k)))

    for i in range(len(k)):
        output_w, output_e, output_alpha, output_beta, output_mu, output_gamma, output_cor, output_res = scca(
            tX, KK, k[i], outdis, debug, sk, oo, ww, ee)

        wa[:, co] = output_w
        e[:, co] = output_e
        resval[co] = output_res
        corval[co] = output_cor

        co += 1

        # dual deflation
        projk[:, i] = KK @ e[:, i]
        tau[:, i] = KK @ projk[:, i]

        P = np.eye(len(KK)) - \
            (np.outer(tau[:, i], tau[:, i])) / (tau[:, i].T @ tau[:, i])
        KK = P.T @ KK @ P

        # primal deflation
        proj[:, i] = tX @ (tX.T @ wa[:, i])
        t[:, i] = tX.T @ proj[:, i]
        tX = tX - tX @ (np.outer(t[:, i], t[:, i])) / (t[:, i].T @ t[:, i])
    print('    ')

    # Primal projection
    P = trainX @ t @ np.linalg.inv(t.T @ t)
    W = proj @ np.linalg.inv(P.T @ proj)

    # can't think of a fancy way to normalise the vectors
    WW = W.copy()
    for i in range(W.shape[1]):
        WW[:, i] = W[:, i] / np.linalg.norm(trainX.T @ W[:, i])
    W = WW

    # Dual Projection
    Z = projk @ np.linalg.inv(np.linalg.inv(tau.T @ tau) @ tau.T @ Kb @ projk)

    ZZ = Z.copy()
    for i in range(Z.shape[1]):
        ZZ[:, i] = Z[:, i] / np.linalg.norm(Kb @ Z[:, i])
    Z = ZZ

    output = {"w": wa,
              "e": e,
              "P": P,
              "dual_tau": tau,
              "primal_tau": t,
              "cor": corval,
              "res": resval,
              "W": W,
              "Z": Z}
    return output
