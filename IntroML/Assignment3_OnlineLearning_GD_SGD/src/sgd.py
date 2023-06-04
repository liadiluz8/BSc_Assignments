#################################
# Your name: Liad Iluz
#################################

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    w_t = np.zeros(data.shape[1])
    index_arr = np.arange(data.shape[0])
    for t in np.arange(1,T+1):  # T times
        eta = eta_0/t
        i = np.random.choice(index_arr)
        x_i, y_i = data[i], labels[i]
        w_tp1 = (1-eta)*w_t+eta*C*y_i*x_i if y_i*np.dot(w_t, x_i)<1 else (1-eta)*w_t
        w_t = w_tp1
    return w_t

def SGD_log(data, labels, eta_0, T, w_t_arr = np.array([])):
    """
    Implements SGD for log loss.
    """
    w_t = np.zeros(data.shape[1])
    index_arr = np.arange(data.shape[0])
    for t in np.arange(1,T+1):  # T times
        if w_t_arr.size != 0: # for 2(c)
            w_t_arr[t-1] = np.linalg.norm(w_t)
        eta = eta_0/t
        i = np.random.choice(index_arr)
        x_i, y_i = data[i], labels[i]
        w_tp1 = w_t + eta*y_i*softmax([0,-y_i*np.dot(w_t,x_i)])[1]*x_i
        w_t = w_tp1
    return w_t

#################################
# Place for additional code
def plot_errors(x, y, x_axis_name, y_axis_name, title):
    plt.title(title)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.xscale('log', base=10)
    plt.plot(x, y)
    plt.show()

def calc_error(data, labels, w):
    err = np.mean(np.dot(w,data.transpose())*labels >= 0)
    return err

def OptimalEta0(train_data, train_labels, validation_data, validation_labels, C, T, eta_rng, SGD):
    runs, j = 10, 0
    err_lst = np.zeros(eta_rng.size)
    for eta_0 in eta_rng:
        err = 0.0
        for i in range(runs):
            if SGD==SGD_hinge:
                w = SGD(train_data, train_labels, C, eta_0, T)  # SGD_hinge
                err += calc_error(validation_data, validation_labels, w) # by hinge-loss
            else:
                w = SGD(train_data, train_labels, eta_0, T) # SGD_log
                err += calc_error(validation_data, validation_labels, w) # by log-loss
        err /= runs
        err_lst[j] = err
        j+=1
    plot_errors(eta_rng, err_lst, "eta0", "error", "Finding optimal eta0")
    return eta_rng[np.argmax(err_lst)]

def OptimalC(train_data, train_labels, validation_data, validation_labels, T,eta_0, C_rng):
    runs, j = 10, 0
    err_lst = np.zeros(C_rng.size)
    for C in C_rng:
        err = 0.0
        for i in range(runs):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            err += calc_error(validation_data, validation_labels, w)
        err /= runs
        err_lst[j] = err
        j+=1
    plot_errors(C_rng, err_lst, "C", "error", "Finding optimal C")
    return C_rng[np.argmax(err_lst)]

def ImageForOptimalValues(train_data, train_labels, eta_0, C, T, SGD):
    if SGD==SGD_hinge:
        w = SGD(train_data, train_labels, C, eta_0, T)
    else:
        w = SGD(train_data, train_labels, eta_0, T)
    im = plt.imshow(np.reshape(w, (28,28)), interpolation='nearest')
    im.axes.xaxis.set_visible(False)
    im.axes.yaxis.set_visible(False)
    plt.title("Best Classifier Image")
    plt.colorbar(im)
    plt.show()
    return w

def Norm_wrt_T(train_data, train_labels, eta_0, T):
    w_t_norms = np.zeros(T)
    SGD_log(train_data, train_labels, eta_0, T, w_t_norms)  # mutate w_t_norms by assignment
    plt.title("The norm of w_t wrt the iteration t")
    plt.xlabel("t")
    plt.ylabel("norm(w_t)")
    plt.plot(np.arange(T), w_t_norms)
    plt.show()
    pass
#################################

def Q1(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    ######################## Ex1 ######################
    # Ex1(a)
    T = 1000
    eta_rng = np.logspace(-5, 2, 100)
    op_eta0 = OptimalEta0(train_data, train_labels, validation_data, validation_labels, 1, T, eta_rng, SGD_hinge)
    print("Best eta0:", op_eta0)
    # Ex1(b)
    C_rng = np.logspace(-5, 5, 100)
    op_C = OptimalC(train_data, train_labels, validation_data, validation_labels, T, op_eta0, C_rng)
    print("Best C:", op_C)
    # Ex1(c)
    T = 20000
    w = ImageForOptimalValues(train_data, train_labels, op_eta0, op_C, T, SGD_hinge)
    # Ex1(d)
    best_test_error = calc_error(test_data, test_labels, w)
    print("Accuracy on the test set:", best_test_error)

def Q2(train_data, train_labels, validation_data, validation_labels, test_data, test_labels):
    ######################## Ex2 ######################
    # Ex2(a)
    T=1000
    eta_rng = np.logspace(-7, -2, 100)
    op_eta0 = OptimalEta0(train_data, train_labels, validation_data, validation_labels, 1, T, eta_rng, SGD_log)
    print("Best eta0:", op_eta0)
    # Ex2(b)
    T = 20000
    w = ImageForOptimalValues(train_data, train_labels, op_eta0, 0, T, SGD_log)
    best_test_error = calc_error(test_data, test_labels, w)
    print("Accuracy on the test set:", best_test_error)
    # Ex2(c)
    Norm_wrt_T(train_data, train_labels, op_eta0, T)

if __name__ == '__main__':
    # Data creation and separation
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    Q1(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)
    Q2(train_data, train_labels, validation_data, validation_labels, test_data, test_labels)