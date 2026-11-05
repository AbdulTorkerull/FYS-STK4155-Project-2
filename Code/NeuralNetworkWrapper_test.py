"""Test file for the NeuralNetRegressor class. The testing is done by comparing the class against results from already esteablished
and trusted packages like Scikit-Learn, TensorFlow/Keras, PyTorch, and seing if they produce similiar enough results"""


from Functions import identity, identity_der, sigmoid, sigmoid_der, mse, mse_der, runge
from NeuralNetworkWrapper import NeuralNetRegressor
from NeuralNetworkClass import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np


#  --- Parameters --- #
lam = 1e-9
penalty = "L2"
lr = 0.01
adaptive_method = "ADAM"
epochs = 500
batch_size= 50
epsilon = 1e-9
seed = 2026

# --- Data --- #
datapoints = 100
x = np.linspace(-1, 1, datapoints).reshape(-1, 1)
y = runge(np.linspace(-1, 1, datapoints)).reshape(-1, 1)


# --- Neural Network structure --- #
network_input_size = 1
layer_output_sizes = [50, 50, 1]
activation_funcs = [sigmoid, sigmoid, identity]
activation_ders = [sigmoid_der, sigmoid_der, identity_der]
cost_fun = mse
cost_der = mse_der


# --- Feed Forward Neural Network direct result --- #
NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der)
NN.optimizer(adaptive_method)
NN.learning_rate(lr)
NN.change_seed(seed)
NN.add_penalty(lam, penalty)
NN.change_seed(seed)
epochs_run, costs, unpenalized_costs = NN.train(x, y, epochs=epochs, M=batch_size, epsilon=epsilon)
y_hat = NN.predict(x)

# --- Feed Forward Neural Network wrapper result --- #
NN_wrapper = NeuralNetRegressor(tuple(layer_output_sizes), activation_funcs=activation_funcs, activation_ders=activation_ders, cost_fun=cost_fun, cost_der=cost_der, optimizer_method=adaptive_method, penalty_method=penalty, gamma=lr, lam=lam, epochs=epochs, M=batch_size, epsilon=epsilon, seed=seed)
NN_wrapper.fit(x, y)
y_hat_wrapper = NN_wrapper.predict(x).reshape(-1, 1)

# --- Compare methods --- #
print("")
print(f"MSE through network directly: {mse(y_hat, y)}")
print(f"MSE through network wrapper:  {mse(y_hat_wrapper, y)}")
print(f"Mean absolute difference between predictions: {np.mean(np.abs(y_hat - y_hat_wrapper))}")
print("")