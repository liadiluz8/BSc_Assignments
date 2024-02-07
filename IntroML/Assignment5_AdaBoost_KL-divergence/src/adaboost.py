#################################
# Your name:Liad Iluz
#################################

from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data
from scipy.special import softmax
np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    n = y_train.size
    
    hypotheses = []
    alpha_vals = []
    
    theta_values = np.unique(X_train)

    Dt = init_distribution(n)
    for t in np.arange(T):
        h_pred, h_index, h_theta, ht = WL(Dt, X_train, y_train, theta_values)
        hypotheses.append((h_pred, h_index, h_theta))
        eps_t = empirical_error_prob(Dt, y_train, ht)
        wt = (1/2)*np.log((1-eps_t)/eps_t)
        alpha_vals.append(wt)
        Dt = softmax(np.log(Dt)+(-1)*wt*y_train*ht)
    
    return (np.array(hypotheses), np.array(alpha_vals))

##############################################
def init_distribution(n):
    return np.full(n,1/n)

def calc_error_h(Dt, X_train, y_train, theta, index):
    pos_pred_err = 0
    neg_pred_err = 0
    
    pos_pred_err = empirical_error_prob(Dt, y_train, calc_ht(1, index, theta, X_train))
    neg_pred_err = empirical_error_prob(Dt, y_train, calc_ht(-1, index, theta, X_train))

    return 2*(pos_pred_err<=neg_pred_err)-1, min(pos_pred_err, neg_pred_err) 

def calc_ht(h_pred, h_index, h_theta, X):
    return 2*(X[:,int(h_index)]<=h_theta)-1 if h_pred==1 else 2*(X[:,int(h_index)]>h_theta)-1

def best_theta_pred_per_index(Dt, X_train, y_train, index, theta_values):
    best_pred = 0
    best_err = 1
    best_theta = -1

    for theta in theta_values:
        pred, err = calc_error_h(Dt, X_train, y_train, theta, index)
        if err < best_err:
            best_pred, best_theta, best_err = (pred, theta, err)
    return best_pred, best_theta, best_err

def WL(Dt, X_train, y_train, theta_values):
    h_pred, h_index, h_theta, best_err = (0, -1, -1, 1)

    for index in np.arange(y_train.size):
        pred, theta, err = best_theta_pred_per_index(Dt, X_train, y_train, index, theta_values)
        if err < best_err:
            h_pred, h_index, h_theta, best_err = (pred, index, theta, err)

    ht = calc_ht(h_pred, h_index, h_theta, X_train)
    
    return h_pred, h_index, h_theta, ht

def empirical_error_prob(Dt, y, ht):
    return np.sum(Dt*(y!=ht))

def classify_X_at_all_t(X, hypotheses, alpha_vals, T):
    summ = np.zeros(X.shape[0])
    classify_T_arr = []

    for t in np.arange(T):
        h_pred, h_index, h_theta = hypotheses[t]
        summ += alpha_vals[t]*calc_ht(h_pred, h_index, h_theta, X)
        classify_T_arr.append(2*(summ>=0)-1)
    
    return np.array(classify_T_arr)

def calc_finel_error(X, y, hypotheses, alpha_vals, T, loss='zo'):
    classify_X_arr = classify_X_at_all_t(X, hypotheses, alpha_vals, T)
    if loss == 'zo':
        return np.array([np.average(classify_X_arr[t]!=y) for t in range(T)])
    elif loss == 'exp':
        mat = classify_X_arr * np.tile(alpha_vals, (classify_X_arr.shape[1], 1)).transpose()
        return np.array([np.average(np.exp(-1*y*np.sum(mat[:t+1,:], axis=0))) for t in range(T)])
    return
    
def drawGraph(title, x, y1, y2, label_x, label_y1, label_y2):
    plt.title(title)
    plt.xlabel(label_x)
    plt.ylabel("Error")
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.legend({label_y1, label_y2})
    plt.show()

def sectionA(X_train, y_train, X_test, y_test, hypotheses, alpha_vals, T):
    train_errors = calc_finel_error(X_train, y_train, hypotheses, alpha_vals, T)
    test_errors = calc_finel_error(X_test, y_test, hypotheses, alpha_vals, T)
    drawGraph("Train and Test Error wrt t", np.arange(T), train_errors, test_errors,"T", "train error", "test error")

def sectionB(hypotheses, T, vocab):
    for t in range(T):  # T<=80
        h_pred, h_index, h_theta = hypotheses[t]
        print("Weak classifier at t={0}:\th_pred={1}\th_index={2}[{3}]\th_theta={4}".format(t,h_pred,h_index,vocab[h_index],h_theta))

def sectionC(X_train, y_train, X_test, y_test, hypotheses, alpha_vals, T):
    train_errors = calc_finel_error(X_train, y_train, hypotheses, alpha_vals, T, 'exp')
    test_errors = calc_finel_error(X_test, y_test, hypotheses, alpha_vals, T, 'exp')
    drawGraph("Train and Test loss exp Error wrt t", np.arange(T), train_errors, test_errors,"T", "train error", "test error")


##############################################


def main():
    T = 80
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    sectionA(X_train, y_train, X_test, y_test, hypotheses, alpha_vals, T)
    sectionB(hypotheses, 10, vocab)
    sectionC(X_train, y_train, X_test, y_test, hypotheses, alpha_vals, T)
    ##############################################

if __name__ == '__main__':
    main()

