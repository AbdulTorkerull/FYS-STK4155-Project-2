import numpy as np 
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der, classification=False):
        self.network_input_size =  network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der
        self.classification = classification

        self.rng = np.random.default_rng(2026)
        self.layers = self._create_layers(network_input_size, layer_output_sizes)
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers]
        self.method = "PLAIN"
        self.penalty = "L2"
        self.lam = 0.0
        self.gamma = 0.001
        self.t = 0

    def _init_state(self):
        self.layers = self._create_layers(self.network_input_size, self.layer_output_sizes)
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers]
        self.t = 0

        return self.layers, self.vrm, self.t


    def _create_layers(self, network_input_size, layer_output_sizes):
        layers = []

        layer_input_size = network_input_size
        for layer_output_size, activation in zip(layer_output_sizes, self.activation_funcs):
            activation_name = getattr(activation, "__name__", "").lower()

            if activation_name in ("relu", "leaky_relu"): 
                scaling = np.sqrt(2/layer_input_size) # Add He scaling to layers using ReLU family as activation function
            else: 
                scaling = np.sqrt(1/layer_input_size) # Add Xavier scaling to layers using any other activation function

            W = self.rng.standard_normal((layer_input_size, layer_output_size))*scaling
            b = np.zeros((1, layer_output_size))
            layers.append((W, b))

            layer_input_size = layer_output_size

        return layers

    def _feed_forward(self, inputs):
        layer_inputs = []
        layer_outputs = []
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)

            z = a @ W + b
            a = activation_func(z)

            layer_outputs.append(z)

        return layer_inputs, layer_outputs, a

    def _adaptive_method_step(self, method, theta, grad, vrm, gamma, t, beta=0.9, beta1=0.9, beta2=0.999, rho=0.99, eps=1e-9):

        if method == "PLAIN":                                   
            return theta - gamma*grad
        
        if method == "MOMENTUM":                                 
            vrm[0] = beta*vrm[0] + gamma*grad

            return theta - vrm[0]
        
        if method == "ADAGRAD":                                  
            vrm[1] = vrm[1] + grad**2

            return theta - (gamma*grad)/np.sqrt(vrm[1] + eps)
        
        if method == "RMSPROP":                                  
            vrm[1] = rho*vrm[1] + (1 - rho)*grad**2 

            return theta - (gamma*grad)/np.sqrt(vrm[1] + eps)
        
        if method == "ADAM":                                    
            vrm[2] = beta1*vrm[2] + (1 - beta1)*grad
            vrm[1] = beta2*vrm[1] + (1 - beta2)*grad**2
            m_hat, r_hat = vrm[2]/(1 - beta1**t), vrm[1]/(1 - beta2**t)

            return theta - (gamma*m_hat)/(np.sqrt(r_hat) + eps)
        
    def _get_minibatches(self, n, M):
        idx = self.rng.permutation(n)

        return [idx[i:i + M] for i in range(0, n, M)]

    def change_seed(self, seed):
        self.rng = np.random.default_rng(seed)

        return seed

    def optimizer(self, method):
        self.method = method

        return self.method

    def learning_rate(self, gamma):
        self.gamma = gamma

        return self.gamma

    def add_penalty(self, lam, penalty):
        self.lam = lam
        self.penalty = penalty

        return self.lam, self.penalty

    def predict(self, inputs):
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = a @ W + b
            a = activation_func(z)

        return a

    def unpenalized_cost(self, inputs, targets):
        final_output = self.predict(inputs)
        
        return self.cost_fun(final_output, targets)

    def cost(self, inputs, targets):
        final_output = self.predict(inputs)
        pen = 0.0

        if self.penalty == "L1":
            pen = self.lam*sum(np.sum(np.abs(W)) for W, _ in self.layers)
        elif self.penalty == "L2":
            pen = self.lam*sum(np.sum(W**2) for W, _ in self.layers)
        elif self.penalty is not None:
            raise ValueError(f"Unknown penalty: {self.penalty!r}")

        return self.cost_fun(final_output, targets) + pen

    def compute_gradient(self, inputs, targets):
        layer_inputs, layer_outputs, final_output = self._feed_forward(inputs)
        layer_grads = [() for _ in self.layers]

        for i in reversed(range(len(self.layers))):
            a, z, activation_der = layer_inputs[i], layer_outputs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                if self.classification is True:
                    dC_dz  = (final_output - targets)/inputs.shape[0]
                else:
                    dC_da  = self.cost_der(final_output, targets)
                    dC_dz = dC_da*activation_der(z)

            else:
                (W, b) = self.layers[i + 1] 
                dC_da = dC_dz @ W.T
                dC_dz = dC_da*activation_der(z)

            (W, b) = self.layers[i]
            dC_db = np.sum(dC_dz, axis=0, keepdims=True)
            dC_dW = a.T @ dC_dz
            
            if self.penalty == "L1":  dC_dW += self.lam*np.sign(W)
            elif self.penalty == "L2": dC_dW += 2*self.lam*W
            elif self.penalty is not None: raise ValueError(f"Unknown penalty: {self.penalty!r}")

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_grads):

        self.layers = [(self._adaptive_method_step(self.method, W, W_g, vrm_W, self.gamma, self.t),
                        self._adaptive_method_step(self.method, b, b_g, vrm_b, self.gamma, self.t))
                        for (W, b), (W_g, b_g), (vrm_W, vrm_b) in zip(self.layers, layer_grads, self.vrm)]

        return self.layers

    def train(self, inputs, targets, epochs=100, M=None, epsilon=1e-6):
        if M is None:
            M = inputs.shape[0]

        self._init_state()
        costs = []
        unpenalized_costs = []

        for epoch in range(1, epochs + 1):
            for idx in self._get_minibatches(inputs.shape[0], M):
                inputs_batch, targets_batch = inputs[idx], targets[idx]

                self.t += 1
                grads = self.compute_gradient(inputs_batch, targets_batch)
                self.layers = self.update_weights(grads)

            costs.append(self.cost(inputs, targets))
            unpenalized_costs.append(self.unpenalized_cost(inputs, targets))

            if epoch > 1 and abs(costs[-2] - costs[-1]) < epsilon:
                break

        return epoch, costs, unpenalized_costs

    def jax_compliant_predict(self, layers, inputs):
        pass

    def jax_gradient(self, inputs, targets):
        pass

if __name__ == "__main__":
    from Functions import linear, linear_der, ReLU, ReLU_der, sigmoid, sigmoid_der, mse, mse_der, softmax, cross_entropy, runge
    from sklearn import datasets
    import time

    """Check if NeuralNetwork class is capable of solving a regression problem by trying to fit the runge function"""
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

    print(f"Regression (Runge function)")
    print(f"Runtime: {runtime}")
    print(f"Epochs: {epochs}")

    plt.title("Regression problem (Runge function)")
    plt.plot(x.ravel(), y_hat.ravel(), label="Fitted Runge function (FFNN)")
    plt.plot(x.ravel(), y.ravel(), label="Runge function")
    plt.legend()
    plt.show()

    """Check if NeuralNetwork class is capable of solving a classification problem by trying to classify Iris dataset"""
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

    print(f"Classification (Iris data)")
    print(f"Runtime: {runtime}")
    print(f"Epochs: {epochs}")

    plt.title("Classification problem (Iris dataset)")
    plt.plot(np.linspace(0, epochs, epochs), costs, label="Cost")
    plt.plot(np.linspace(0, epochs, epochs), unpenalized_costs, label="Unpenalized Cost")
    plt.legend()
    plt.show()