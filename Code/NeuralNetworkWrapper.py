import numpy as np
from NeuralNetworkClass import NeuralNetwork
from sklearn.base import BaseEstimator, RegressorMixin

class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, layer_output_sizes=(50, 1), activation_funcs=None, activation_ders=None, cost_fun=None, cost_der=None, optimizer_method="ADAM", penalty_method="L2", gamma=0.01, lam=0.0, epochs=1000, M=10, epsilon=1e-9, seed=2026):
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.optimizer_method = optimizer_method
        self.penalty_method = penalty_method
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.M = M
        self.epsilon = epsilon
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        Y = y.reshape(1, -1)

        network_input_size = X.shape[1]
        layer_output_sizes = list(self.layer_output_sizes)
        self.NN = NeuralNetwork(network_input_size, layer_output_sizes, self.activation_funcs, self.activation_ders, self.cost_fun, self.cost_der)

        self.NN.optimizer(self.optimizer_method)
        self.NN.add_penalty(self.lam, self.penalty_method)
        self.NN.learning_rate(self.gamma)
        self.NN.change_seed(self.seed)

        self.n_epochs_, self.cost_history_, self.unpenalized_history_ = self.NN.train(X.T, Y, epochs=self.epochs, M=self.M, epsilon=self.epsilon)

        return self

    def predict(self, X):
        return np.asarray(self.NN.predict(np.asarray(X, dtype=float).T)).ravel()
