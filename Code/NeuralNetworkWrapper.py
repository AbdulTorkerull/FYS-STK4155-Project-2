"""File NeuralNetRegressor class, that is wrapper class for the NeuralNetwork class to make it compatiple with scikit.
This is required in order to do cross validation and grid search for the project"""


import numpy as np
from NeuralNetworkClass import NeuralNetwork
from sklearn.base import BaseEstimator, RegressorMixin


class NeuralNetRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper around the NeuralNetwork class. This makes it possible to use the FFNN with scikit-learn tools
    like cross validation and grid search. All hyperparameters are stored as attributes in __init__ as scikit-learn requires."""

    def __init__(self, layer_output_sizes=(50, 1), activation_funcs=None, activation_ders=None, cost_fun=None, cost_der=None, optimizer_method="ADAM", penalty_method="L2", gamma=0.01, lam=0.0, epochs=1000, M=10, epsilon=1e-9, seed=2026):
        self.layer_output_sizes = layer_output_sizes # Output sizes of each layer, the last one being the output of the network (Tuple of Integers)
        self.activation_funcs = activation_funcs # Activation function of each output layer (List of Functions)
        self.activation_ders = activation_ders # Activation derivatives of each output layer (List of Functions)
        self.cost_fun = cost_fun # Cost function used in the final layer (Function)
        self.cost_der = cost_der # Cost derivative used in the final layer (Function)
        self.optimizer_method = optimizer_method # Adaptive method used for optimizing cost function
        self.penalty_method = penalty_method # Penalty method used in the cost function (L1 or L2)
        self.gamma = gamma # Learning rate
        self.lam = lam # Penalty strength
        self.epochs = epochs # Maximum number of epochs in training
        self.M = M # Minibatch size
        self.epsilon = epsilon # Tolerance for stopping training early when the cost has converged
        self.seed = seed # Seed used for all randomness in the network

    def fit(self, X, y):
        """Method for training the network on a matrix of inputs X with shape (samples) x (features) and targets y. Returns self as scikit-learn requires"""

        X = np.asarray(X, dtype=float) # Make sure inputs are a float array
        y = np.asarray(y, dtype=float).ravel() # Flatten targets into a vector of floats
        Y = y.reshape(-1, 1) # Targets as a column vector with shape (samples) x (1), which is what the network expects

        network_input_size = X.shape[1] # Number of features is the input size of the network
        layer_output_sizes = list(self.layer_output_sizes) # Convert from tuple to list
        self.NN = NeuralNetwork(network_input_size, layer_output_sizes, self.activation_funcs, self.activation_ders, self.cost_fun, self.cost_der) # Create the network

        self.NN.optimizer(self.optimizer_method) # Set optimizer method
        self.NN.add_penalty(self.lam, self.penalty_method) # Set penalty strength and penalty method
        self.NN.learning_rate(self.gamma) # Set learning rate
        self.NN.change_seed(self.seed) # Set seed

        self.n_epochs_, self.cost_history_, self.unpenalized_history_ = self.NN.train(X, Y, epochs=self.epochs, M=self.M, epsilon=self.epsilon) # Train and store number of epochs run and the cost histories, trailing underscore marks attributes set in fit

        return self

    def predict(self, X):
        """Method for predicting targets from a matrix of inputs X with shape (samples) x (features). Returns a vector with one prediction per sample"""

        return np.asarray(self.NN.predict(np.asarray(X, dtype=float))).ravel() # Predict with the trained network and flatten the output into a vector


if __name__ == "__main__":
    pass