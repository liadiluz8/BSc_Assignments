from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np

# Load dataset
mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

# Define the training and the test set
rng = 70000
samples = 11000
train_samples = 10000
idx = np.random.RandomState(0).choice(rng, samples)
train = data[idx[:train_samples], :].astype(int)
train_labels = labels[idx[:train_samples]]
test = data[idx[train_samples:], :].astype(int)
test_labels = labels[idx[train_samples:]]

# k_NN algorithm implementation
def k_NN(train, train_labels, query_img, k):
    """
    k_NN function
    -------------
    Implementation of k Nearest Neighbor algorithm

    Parameters:
    -----------
    train: np.array
    A set of imegaes (as metrix) which the k_NN algorithm train will tarin on

    train_labels: np.array
    A vector of labels (as array) corresponding the train images

    query_img: np.array
    An image (as array) which the k_NN algorithm will predict on

    k: int
    An integer number as parameter of k_NN algorithm
    Assume 0<k<train.size
    
    Retern value:
    -------------
    A number of label which is the prediction of k_NN algoritm
    on the query based on the other parameters
    """

    distances = np.asarray([np.linalg.norm(img - query_img) for img in train])
    k_min_indices = np.argpartition(distances, k-1)[:k]
    k_labels = train_labels[k_min_indices].astype(np.int64)
    predict = np.bincount(k_labels).argmax() # predict=common lable

    return predict

## Runs
## Section b
# k = 10
# n = 1000
# train_n = train[:n, :] # using onle the first n training images
# train_labels_n = train_labels[:n] # using onle the first n training images
# test_pred = np.asarray([k_NN(train_n, train_labels_n, test[i], k) for i in range(test.shape[0])])
# accuracy = accuracy_score(test_labels.astype(np.int64), test_pred)
# print("Accuracy", accuracy)

## Section c
# n = 1000
# train_n = train[:n, :] # using onle the first n training images
# train_labels_n = train_labels[:n] # using onle the first n training images
# accuracy_list = np.zeros(100)
# for k in np.arange(1,101):  # Run test for k from 1 to 100 and get each accuracy
#     test_pred = np.asarray([k_NN(train_n, train_labels_n, test[i], k) for i in range(test.shape[0])])
#     accuracy = accuracy_score(test_labels.astype(np.int64), test_pred)
#     accuracy_list[k-1] = accuracy

## plot accuracy as a function of k
# plt.title("Ex2(b) : The accuracy as a function of k, k=1,...,100")
# plt.xlabel("k")
# plt.ylabel("accuracy")
# plt.plot(np.arange(1,101), np.asarray(accuracy_list))
# plt.show()

# # Section d
# k = 1
# accuracy_list = np.zeros(50)
# i=0
# for n in np.arange(100,5001, 100):  # Run test for n from 100 to 5000 with gap of 100 and get each accuracy
#     train_n = train[:n, :] # using onle the first n training images
#     train_labels_n = train_labels[:n] # using onle the first n training images
#     test_pred = np.asarray([k_NN(train_n, train_labels_n, test[i], k) for i in range(test.shape[0])])
#     accuracy = accuracy_score(test_labels.astype(np.int64), test_pred)
#     accuracy_list[i] = accuracy
#     i+=1

# # plot accuracy as a function of k
# plt.title("Ex2(d) : The accuracy as a function of n, n=100,200,...,5000")
# plt.xlabel("n")
# plt.ylabel("accuracy")
# plt.plot(np.arange(100,5001, 100), np.asarray(accuracy_list))
# plt.show()