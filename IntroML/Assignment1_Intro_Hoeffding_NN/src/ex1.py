import numpy as np
from matplotlib import pyplot as plt

N = 200000
n = 20
# Generate N*n matrix of samples from Bernoulli(1/2)
bin_mat = np.random.binomial(n=1,  p=0.5, size=N*n).reshape((N,n))
# Compute the empirical mean for each row
emp_min = (1/n)*np.sum(bin_mat, axis=1)
# 50 values of [0,1]
epsilon = np.linspace(0,1,50)
# Calculate the empirical prob. that |emp_min - 1/2|>epsilon
emp_prob = np.asarray([np.sum(abs(emp_min - 1/2)>e)/N for e in epsilon])
# Ploting the empirical graph and the Hoeffding bound graph of this prob.
plt.title("Ex1 : Empirical test and Hoeffding bound of N=200000 samples of Bernouli(1/2) where n=20")
plt.xlabel("epsilon")
plt.ylabel("The probability that |X - 1/2|>epsilon")
plt.plot(epsilon, emp_prob, 'o')
hoeff = 2*np.exp(-2*n*(epsilon**2))
plt.plot(epsilon,hoeff)
plt.legend({'empirical test','Hoeffding bound'})
plt.show()
