import backprop_data

import backprop_network



training_data, test_data = backprop_data.load(train_size=5000,test_size=1000)

# q1(b)
net = backprop_network.Network([784, 40, 10])
net.SGD_Qb(training_data, epochs=30, mini_batch_size=10, learning_rate_range=[0.001,0.01,0.1,1,10,100], test_data=test_data)

# q1(c)
# net = backprop_network.Network([784, 40, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# q1(d)
# net = backprop_network.Network([784, 400, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=4, learning_rate=0.1, test_data=test_data)
