"""File containing the Neural Network class that will be used to solve most of the exercises in the project. 
It also contains two examples, one where its used to solve a regression problem and one where its used to
to solve a classification problem"""

import numpy as np 
import matplotlib.pyplot as plt


class NeuralNetwork:
    """Feed Forward Neural Network (FFNN) class for solving both regression and classification problems. The class can be used with any number
    of hidden layers, nodes per layer, activation functions and cost function."""

    def __init__(self, network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der, classification=False):
        self.network_input_size =  network_input_size # Network input size (Integer)
        self.layer_output_sizes = layer_output_sizes # Network output sizes (List of Integers)
        self.activation_funcs = activation_funcs # Activation function of each output layer (List of Functions)
        self.activation_ders = activation_ders # Activation derivatives of each output layer (List of Functions)
        self.cost_fun = cost_fun # Cost function used in the final layer (Function)
        self.cost_der = cost_der # Cost derivative used in the final layer (Function)
        self.classification = classification # Tells class wether its solvign a regression or classification problem

        self.rng = np.random.default_rng(2026) # RNG used for all randomness in the class
        self.layers = self._create_layers(network_input_size, layer_output_sizes) # Layers containing all the nodes of the FFNN
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers] # v,r and m values used in the adaptive methods
        self.method = "PLAIN" # Adaptive method used for optimizing cost function
        self.penalty = "L2" # Penalty method used in the cost function
        self.lam = 0.0 # Penalty streangth
        self.gamma = 0.001 # Learning rate
        self.t = 0 # Time step used in adaptive method

    def _init_state(self):
        """Internal method for taking the neuralnetwork back into its initial state before training"""

        self.layers = self._create_layers(self.network_input_size, self.layer_output_sizes) # Restart the layers
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers] # Restart the vrm variables used in the adaptive methods
        self.t = 0 # Restart timestep t used in adaptive methods

        return self.layers, self.vrm, self.t


    def _create_layers(self, network_input_size, layer_output_sizes):
        """Internal function for creating list of weights and biases. These weights and biases are the variables that get trained and optimized 
        to solve the regression/classification problem"""

        layers = [] # List for storing the weight matrices and bias vectors of each layer

        layer_input_size = network_input_size # Input size of current layer
        for layer_output_size, activation in zip(layer_output_sizes, self.activation_funcs): # Iterate over output layers and activation functions
            activation_name = getattr(activation, "__name__", "").lower() # Aquire name of activation function for if-statemt usage

            if activation_name in ("relu", "leaky_relu"): 
                scaling = np.sqrt(2/layer_input_size) # Add He scaling to layers using ReLU family as activation function
            else: 
                scaling = np.sqrt(1/layer_input_size) # Add Xavier scaling to layers using any other activation function

            W = self.rng.standard_normal((layer_input_size, layer_output_size))*scaling # Create weight matrix with shape (current input size) x (current output size)
            b = np.zeros((1, layer_output_size)) # Create bias vector with shape (1) x (current output size)
            layers.append((W, b)) # append both into layers as a tuple

            layer_input_size = layer_output_size # Change current input size to be the next output size

        return layers

    def _feed_forward(self, inputs):
        """Internal method for doing a singular feed forward step through the FFNN using a matrix of inputs (samples) x (features)"""

        layer_inputs = [] # List for storing input to all layers
        layer_outputs = [] # List for storing output of all layers
        a = inputs # a is input to current layer

        for (W, b), activation_func in zip(self.layers, self.activation_funcs): # Iterate over layers and activation functions
            layer_inputs.append(a) 

            z = a @ W + b # z is ouput to current layer
            a = activation_func(z)

            layer_outputs.append(z)

        return layer_inputs, layer_outputs, a

    def _adaptive_method_step(self, method, theta, grad, vrm, gamma, t, beta=0.9, beta1=0.9, beta2=0.999, rho=0.99, eps=1e-9):
        """Internal method containing singular gradien descent step using one of a chosen methods;
        PLAIN, MOMENTUM, ADAGRAD, RMSPROP and ADAM"""

        if method == "PLAIN":                                   
            return theta - gamma*grad # Plain gradient descent formula
        
        if method == "MOMENTUM":                                 
            vrm[0] = beta*vrm[0] + gamma*grad # Compute "v"

            return theta - vrm[0] # Gradient descent with Momentum formula
        
        if method == "ADAGRAD":                                  
            vrm[1] = vrm[1] + grad**2 # Compute "r"

            return theta - (gamma*grad)/np.sqrt(vrm[1] + eps) # Gradient descent with AdaGrad formula
        
        if method == "RMSPROP":                                  
            vrm[1] = rho*vrm[1] + (1 - rho)*grad**2 # Compute "r"

            return theta - (gamma*grad)/np.sqrt(vrm[1] + eps) # Gradient descent with RMSprop formula
        
        if method == "ADAM":                                    
            vrm[2] = beta1*vrm[2] + (1 - beta1)*grad # Compute "m"
            vrm[1] = beta2*vrm[1] + (1 - beta2)*grad**2 # Compute "r"
            m_hat, r_hat = vrm[2]/(1 - beta1**t), vrm[1]/(1 - beta2**t) # Normalize "r" and "m"

            return theta - (gamma*m_hat)/(np.sqrt(r_hat) + eps) # Gradient descent with Adam formula
        
    def _get_minibatches(self, n, M):
        """Internal method for getting a minibatch of size M from an input with size n"""

        idx = self.rng.permutation(n) # Shuffle the inputs

        return [idx[i:i + M] for i in range(0, n, M)]

    def change_seed(self, seed):
        """Method for changing seed"""

        self.rng = np.random.default_rng(seed)

        return seed

    def optimizer(self, method):
        """Method for changing optimizer method. Valid methods are;
        PLAIN, MOMENTUM, ADAGRAD, RMSPROP and ADAM"""

        self.method = method

        return self.method

    def learning_rate(self, gamma):
        """Method for changing learning rate"""

        self.gamma = gamma

        return self.gamma

    def add_penalty(self, lam, penalty):
        """Method for changing penalty streangth and penalty method.
        Valdi penalty methods are; L1, L2"""

        self.lam = lam
        self.penalty = penalty

        return self.lam, self.penalty

    def predict(self, inputs):
        """Method for predicting the output of the FFNN using a matrix of inputs (samples) x (features). Does the same as _feed_forward,
        but without storing the layer inputs and outputs since they are only needed for backpropagation"""

        a = inputs # a is input to current layer

        for (W, b), activation_func in zip(self.layers, self.activation_funcs): # Iterate over layers and activation functions
            z = a @ W + b # z is output to current layer before activation
            a = activation_func(z) # a is now the input to the next layer

        return a # Output of the final layer, i.e. the prediction of the network

    def unpenalized_cost(self, inputs, targets):
        """Method for computing the cost of the network on some data without the penalty term. Useful for comparing 
        models trained with different penalty strengths on equal terms"""

        final_output = self.predict(inputs) # Prediction of the network
        
        return self.cost_fun(final_output, targets) # Cost function evaluated on prediction and targets

    def cost(self, inputs, targets):
        """Method for computing the full cost of the network on some data, which is the cost function plus the penalty term.
        The penalty is applied to the weights only, not the biases"""

        final_output = self.predict(inputs) # Prediction of the network
        pen = 0.0 # Penalty term, stays zero if no penalty is chosen

        if self.penalty == "L1":
            pen = self.lam*sum(np.sum(np.abs(W)) for W, _ in self.layers) # L1 penalty: lambda times sum of absolute values of all weights
        elif self.penalty == "L2":
            pen = self.lam*sum(np.sum(W**2) for W, _ in self.layers) # L2 penalty: lambda times sum of squared values of all weights
        elif self.penalty is not None:
            raise ValueError(f"Unknown penalty: {self.penalty!r}") # Only L1, L2 or None are valid penalties

        return self.cost_fun(final_output, targets) + pen # Cost function plus penalty

    def compute_gradient(self, inputs, targets):
        """Method for computing the gradient of the cost function with respect to the weights and biases of every layer using
        backpropagation. Returns a list with a tuple (dC_dW, dC_db) for each layer"""

        layer_inputs, layer_outputs, final_output = self._feed_forward(inputs) # Feed forward and store what is needed for backpropagation
        layer_grads = [() for _ in self.layers] # List for storing the gradients of each layer

        for i in reversed(range(len(self.layers))): # Iterate over layers backwards, from output layer to first layer
            a, z, activation_der = layer_inputs[i], layer_outputs[i], self.activation_ders[i] # Input, output (before activation) and activation derivative of current layer

            if i == len(self.layers) - 1: # Final layer, start of the backpropagation
                if self.classification is True:
                    dC_dz  = (final_output - targets)/inputs.shape[0] # Classification: cost derivative and activation derivative combine into (a - y), averaged over the samples
                else:
                    dC_da  = self.cost_der(final_output, targets) # Regression: derivative of the cost with respect to the output of the network
                    dC_dz = dC_da*activation_der(z) # Chain rule through the activation function

            else: # Hidden layers, propagate the error backwards
                (W, b) = self.layers[i + 1] # Weights of the layer after the current one
                dC_da = dC_dz @ W.T # Error from the next layer propagated back to the output of current layer
                dC_dz = dC_da*activation_der(z) # Chain rule through the activation function

            (W, b) = self.layers[i] # Weights and biases of current layer
            dC_db = np.sum(dC_dz, axis=0, keepdims=True) # Bias gradient, summed over all samples
            dC_dW = a.T @ dC_dz # Weight gradient, input to the layer times the error
            
            if self.penalty == "L1":  dC_dW += self.lam*np.sign(W) # Gradient of the L1 penalty added to the weight gradient
            elif self.penalty == "L2": dC_dW += 2*self.lam*W # Gradient of the L2 penalty added to the weight gradient
            elif self.penalty is not None: raise ValueError(f"Unknown penalty: {self.penalty!r}") # Only L1, L2 or None are valid penalties

            layer_grads[i] = (dC_dW, dC_db) # Store the gradients of current layer

        return layer_grads

    def update_weights(self, layer_grads):
        """Method for updating the weights and biases of all layers with one step of the chosen optimizer, using the 
        gradients computed in compute_gradient"""

        self.layers = [(self._adaptive_method_step(self.method, W, W_g, vrm_W, self.gamma, self.t), # Updated weights of the layer
                        self._adaptive_method_step(self.method, b, b_g, vrm_b, self.gamma, self.t)) # Updated biases of the layer
                        for (W, b), (W_g, b_g), (vrm_W, vrm_b) in zip(self.layers, layer_grads, self.vrm)] # Iterate over layers, their gradients and their v,r,m values

        return self.layers

    def train(self, inputs, targets, epochs=100, M=None, epsilon=1e-6):
        """Method for training the FFNN with minibatch gradient descent. M is the minibatch size, where M=None gives full batch gradient descent.
        Training stops early if the cost changes less than epsilon between two epochs. Returns the number of epochs run, the cost per epoch 
        and the unpenalized cost per epoch"""

        if M is None:
            M = inputs.shape[0] # No minibatch size given, use all samples in every step

        self._init_state() # Always start training from a fresh network
        costs = [] # List for storing the cost after each epoch
        unpenalized_costs = [] # List for storing the cost without penalty after each epoch

        for epoch in range(1, epochs + 1): # Iterate over epochs
            for idx in self._get_minibatches(inputs.shape[0], M): # Iterate over shuffled minibatches
                inputs_batch, targets_batch = inputs[idx], targets[idx] # Pick out the samples in the current minibatch

                self.t += 1 # Increase time step, needed by Adam and must start at 1
                grads = self.compute_gradient(inputs_batch, targets_batch) # Gradients of current minibatch
                self.layers = self.update_weights(grads) # Take a gradient descent step

            costs.append(self.cost(inputs, targets)) # Cost on the full dataset after the epoch
            unpenalized_costs.append(self.unpenalized_cost(inputs, targets)) # Cost without penalty on the full dataset after the epoch

            if epoch > 1 and abs(costs[-2] - costs[-1]) < epsilon: # Stop if the cost has converged
                break

        return epoch, costs, unpenalized_costs
    

if __name__ == "__main__":
    from Functions import linear, linear_der, ReLU, ReLU_der, sigmoid, sigmoid_der, mse, mse_der, softmax, cross_entropy, runge
    from sklearn import datasets
    import time


    """Example of NeuralNetwork class solving a regression problem with the Runge function"""
    datapoints = 100
    network_input_size = 1
    layer_output_sizes = [32, 16, 32, 16, 32, 1]
    activation_funcs = [sigmoid, ReLU, ReLU, sigmoid, sigmoid, linear]
    activation_ders = [sigmoid_der, ReLU_der, ReLU_der, sigmoid_der, sigmoid_der, linear_der]
    cost_fun = mse
    cost_der = mse_der

    x = np.linspace(-1, 1, datapoints).reshape(-1, 1)
    y = runge(np.linspace(-1, 1, datapoints)).reshape(-1, 1)

    t0 = time.perf_counter()
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der)
    NN.optimizer("ADAM")
    NN.learning_rate(0.01)
    NN.add_penalty(1e-5, "L2")
    epochs, costs, unpenalized_costs = NN.train(x, y, epochs=500, M=50)
    y_hat = NN.predict(x)
    t1 = time.perf_counter()
    runtime = t1 - t0

    print("")
    print(f"Regression (Runge function)")
    print(f"Runtime: {runtime}")
    print(f"Epochs: {epochs}")
    print("")

    plt.title("Regression problem (Runge function)")
    plt.plot(x.ravel(), y_hat.ravel(), label="Fitted Runge function (FFNN)")
    plt.plot(x.ravel(), y.ravel(), label="Runge function")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()


    """Example of NeuralNetwork class solving a classification problem using the Iris dataset"""
    iris = datasets.load_iris()

    inputs = iris.data
    targets = np.zeros((len(iris.data), 3))
    for i, t in enumerate(iris.target):
        targets[i, t] = 1

    network_input_size = 4
    layer_output_sizes = [6, 8, 3]
    activation_funcs = [sigmoid, ReLU, softmax]
    activation_ders = [sigmoid_der, ReLU_der, None]
    cost_fun = cross_entropy
    cost_der = None

    t0 = time.perf_counter()
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der, classification=True)
    NN.optimizer("ADAM")
    NN.learning_rate(0.01)
    NN.add_penalty(1e-3, "L2")
    epochs, costs, unpenalized_costs = NN.train(inputs, targets, epochs=500, M=None)
    t1 = time.perf_counter()
    runtime = t1 - t0

    print("")
    print(f"Classification (Iris data)")
    print(f"Runtime: {runtime}")
    print(f"Epochs: {epochs}")
    print("")

    plt.title("Classification problem (Iris dataset)")
    plt.plot(np.linspace(0, epochs, epochs), costs, label="Total Cost")
    plt.plot(np.linspace(0, epochs, epochs), unpenalized_costs, label="Unpenalized Cost")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.show()