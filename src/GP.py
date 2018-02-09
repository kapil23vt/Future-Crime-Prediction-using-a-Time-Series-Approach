import numpy as np
import GPy as gpy
import GPy.kern
import time
import util
import plot
import matplotlib.pyplot as plt
from pandas import Series, DataFrame, Panel
from sklearn.metrics import mean_squared_error
from math import sqrt

def ker_se(x, y, l, horz=1.0):
    '''
    Compute the kernel matrix
    Use square exponential by default
    '''

    n = np.shape(x)[0]
    m = np.shape(y)[0]

    t = np.reshape(x, (np.shape(x)[0], 1, np.shape(x)[1]))
    s = np.reshape(y, (1, np.shape(y)[0], np.shape(y)[1]))

    # tile across columns
    cols = np.tile(t, (1, m, 1))
    # tile across rows
    rows = np.tile(s, (n, 1, 1))
    # get the differences and vectorize
    diff_vec = np.reshape(cols - rows, (n * m, np.shape(t)[2]))

    M = np.diag(l)

    # use multiply and sum to calculate matrix product
    s = np.multiply(-.5, np.sum(np.multiply(diff_vec,
                                            np.transpose(np.dot(M, np.transpose(diff_vec)))), axis=1))
    se = np.reshape(np.multiply(horz, np.exp(s)), (n, m))

    return se


def GaussianProcess(train, train_t, test, test_t, l,
                    horz, sig_eps, predict=True, rmse=True, ker='se'):
    '''
    Given the split data and parameters, train the GP with the specified kernel
    and return the specified results.
    '''
    # Try to be memory efficient by deleting data after use!
    if ker == 'se':
        ker_fun = ker_se

    ker1 = ker_fun(train, train, l, horz)
    L = np.linalg.cholesky(
        ker1 + np.multiply(sig_eps, np.identity(np.shape(ker1)[0])))

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, train_t))

    # Only do this if we request the predictions or rmse
    ret = []
    if predict or rmse:
        ker2 = ker_fun(train, test, l, horz)
        preds = np.dot(np.transpose(ker2), alpha)
        del ker2
        ret.append(preds)

    # Only if we request the rmse
    if rmse:
        npreds = preds / float(preds.sum())
        ntest_t = test_t / float(test_t.sum())
        rmse_val = util.rmse(npreds, ntest_t)
        ret.append(rmse_val)

    # Calculate the marginal likelihood
    likelihood = -.5 * np.dot(np.transpose(train_t), alpha) - np.sum(
        np.log(np.diagonal(L))) - np.shape(ker1)[0] / 2 * np.log(2 * np.pi)
    ret.append(likelihood)

    del alpha
    del L
    del ker1

    return tuple(ret)

def run_gp(good_data, buckets, l, horz, sig_eps_f, logTransform,
           file_prefix, city, GPy=False):
    '''
    Runs our typical GP training process.
    '''
    # Split as specified by the user
    # default is 15, logSpace=True')
    start = time.clock()
    data = util.createBuckets(
        good_data, n_buckets=buckets, logSpace=logTransform)
    train, train_t, test, test_t = util.split(data, 0)
    end = time.clock()
    
    # Calculate sig_eps
    start = time.clock()
    sig_eps = train_t.std()
    
    
    # Run the gaussian process
    predictions, rmse, likelihood = GaussianProcess(
        train, train_t, test, test_t, l, horz, sig_eps)
    end = time.clock()
    

    if logTransform:
        test_t = np.exp(test_t)
        predictions = np.exp(predictions)
    
    
    # Contatenate new test matrix -- this is the expected input.
    
    X_test = np.zeros((test.shape[0], test.shape[1] + 1)).astype(int)
    X_test[:, :-1] = test
    X_test[:, -1] = test_t
    
    
    # Repeat the process with Gpy
    start = time.clock()
    kern = gpy.kern.RBF(input_dim=3, variance=35.053269286, lengthscale=1.46975091813)
    train_t = train_t.reshape((train_t.shape[0], 1))
    m = gpy.models.GPRegression(train, train_t, kern)
    #m.constrain_positive('.*rbf_variance')
    #m.constrain_bounded('.*lengthscale',1.,10. )
    #m.constrain_fixed('.*noise',0.0025)
    #m.optimize(messages=True)
    #return
    m.Gaussian_noise.variance.constrain_fixed(38.190072092)
    end = time.clock()

    print ("Finished training GPy in {}...".format(end - start))

    start = time.clock()
    predictions_optimal = m.predict(test)[0].reshape(
        (test_t.shape[0]))
    end = time.clock()
    print ("Finished GPy predictions in {}...".format(end - start))

    plot.plotDistribution(predictions_optimal, test_t, city, buckets,
                          process='GPy' + file_prefix)
    print ("Finished GPy Distribution Plots...")
    
    if logTransform:
        test_t = np.exp(test_t)
        predictions_optimal = np.exp(predictions_optimal)
