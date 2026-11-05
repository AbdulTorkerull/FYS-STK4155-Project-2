"""Test file for the NeuralNetwork class. The testing is done by comparing the class against already esteablished and 
trusted packages like Scikit-Learn, TensorFlow/Keras, PyTorch, and seing if the produce similiar enough results"""


from Functions import identity, identity_der, ReLU, ReLU_der, sigmoid, sigmoid_der, mse, mse_der, softmax, cross_entropy, runge
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from NeuralNetworkClass import NeuralNetwork
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import warnings


# --- Ignore SK-learn warning --- #
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#  --- Parameters --- #
lam = 1e-9
penalty = "L2"
lr = 0.01
adaptive_method = "ADAM"
epochs = 5000
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


# --- Self written Feed Forward Neural Network --- #
NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der)
NN.optimizer(adaptive_method)
NN.learning_rate(lr)
NN.change_seed(seed)
NN.add_penalty(lam, penalty)
NN.change_seed(seed)
epochs_run, costs, unpenalized_costs = NN.train(x, y, epochs=epochs, M=batch_size, epsilon=epsilon)
y_hat = NN.predict(x)

# --- SK-learns Feed Forward Neural Network --- #
NN_sklearn = MLPRegressor(
    hidden_layer_sizes=tuple(layer_output_sizes[:-1]),
    activation="logistic",
    solver=adaptive_method.lower(),
    learning_rate_init=lr,
    alpha=lam*batch_size,
    batch_size=batch_size,
    max_iter=epochs,
    tol=epsilon,
    n_iter_no_change=epochs,
    random_state=seed)

NN_sklearn.fit(x, y.ravel())
y_hat_sklearn = NN_sklearn.predict(x).reshape(-1, 1)

# --- Compare methods --- #
print("")
print(f"MSE own network: {mse(y_hat, y)}")
print(f"MSE sklearn:     {mse(y_hat_sklearn, y)}")
print(f"Mean absolute difference between predictions: {np.mean(np.abs(y_hat - y_hat_sklearn))}")
print("")

# --- Plot mothods --- #
plt.title("Regression problem (Runge function)")
plt.plot(x.ravel(), y_hat.ravel(), linestyle=":", label="Fitted Runge function (own FFNN class)")
plt.plot(x.ravel(), y_hat_sklearn.ravel(), linestyle="--", label="Fitted Runge function (SK-learn)")
plt.plot(x.ravel(), y.ravel(), linestyle="-", label="Runge function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()

