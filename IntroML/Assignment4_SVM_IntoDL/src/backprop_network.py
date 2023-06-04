"""

backprop_network.py

"""


import random
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))

    def plot_accuracy_across_epochs_of_learning_rates(self, acc_list, epochs_range, learning_rate_range, plt_titles, y_labels):        
        fig, axs = plt.subplots(1, len(acc_list))
        fig.suptitle('Q1(b)')
        
        for acc_across_epochs,j in zip(acc_list,range(len(acc_list))):
            axs[j].set_title(plt_titles[j])
            i = 0
            for rate, acc in zip(learning_rate_range, acc_across_epochs):
                axs[j].plot(epochs_range, acc, label="rate = {}".format(rate))
                i+=1
        
        for ax, y_label in zip(axs.flat,y_labels):
            ax.set(xlabel='epochs', ylabel=y_label)

        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        fig.show()

    def SGD_Qb(self, training_data, epochs, mini_batch_size, learning_rate_range,
            test_data):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  """

        print("Initial test accuracy: {0}".format(self.one_label_accuracy(test_data)))

        n = len(training_data)
        training_acc_rates = []
        test_acc_rates = []
        training_loss_rates = []

        for learning_rate in learning_rate_range:
            print('lr =', learning_rate)
            training_acc = []
            test_acc = []
            training_loss = []

            for j in range(epochs):
                random.shuffle(training_data)
                
                mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
                
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, learning_rate)

                training_acc.append(self.one_hot_accuracy(training_data))
                test_acc.append(self.one_label_accuracy(test_data))
                training_loss.append(self.loss(training_data))
                # print ("Epoch {0} test accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
            
            training_acc_rates.append(training_acc)
            test_acc_rates.append(test_acc)
            training_loss_rates.append(training_loss)
        print(1)
        self.plot_accuracy_across_epochs_of_learning_rates([training_acc_rates, test_acc_rates, training_loss_rates], 
            np.arange(30), learning_rate_range, plt_titles=['Training Accuracy wrt epoch','Test Accuracy wrt epoch','Training Loss wrt epoch'], 
            y_labels=['Training Accuracy','Test Accuracy','Training Loss'])


    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``."""

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """The function receives as input a 784 dimensional 
        vector x and a one-hot vector y.
        The function should return a tuple of two lists (db, dw) 
        as described in the assignment pdf. """

        dw = [] # Derivatives of the loss wrt the weights metrices
        db = [] # Derivatives of the loss wrt the biases

        # Forward
        v = self.network_output_before_softmax(x)
        z = [relu(v) for v in v]
        z.insert(0, np.copy(x))
        
        # Backward
        db_L = self.loss_derivative_wr_output_activations(v[-1], y)
        db.append(db_L)
        
        for l in range(self.num_layers - 2, 0, -1):
            db_l = np.dot(self.weights[l].transpose() ,db[0]) * relu_derivative(v[l - 1])
            db.insert(0, db_l)

        for l in range(self.num_layers - 1):
            dw_l = np.dot(db[l], z[l].transpose())
            dw.append(dw_l)

        return (db,dw)


    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)[-1]), y) # [-1] is my change
         for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results)/float(len(data))


    def one_hot_accuracy(self,data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [(np.argmax(self.network_output_before_softmax(x)[-1]), np.argmax(y))  # [-1] is my change
                          for (x, y) in data]

        return sum(int(x == y) for (x, y) in output_results) / float(len(data))


    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""

        v_array = []
        layer = 0

        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
                v_array.append(x)
            else:
                x = np.dot(w, x) + b
                v_array.append(x)
                x = relu(x)
            
            layer += 1

        return v_array


    def loss(self, data):
        """Return the CE loss of the network on the data"""
        loss_list = []

        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)[-1]   # [-1] is my change
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(),y).flatten()[0])

        return sum(loss_list) / float(len(data))


    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""
        return softmax(output_activations)


    def loss_derivative_wr_output_activations(self, output_activations, y):
        """ return derivative of loss with respect to the output activations before softmax """
        return self.output_softmax(output_activations) - y



def relu(z):
    """return the relu function in a vector z."""
    return np.maximum(0,z)


def relu_derivative(z):
    """return the derivative of the relu function on vector z."""
    return 1*(relu(z)!=0)

