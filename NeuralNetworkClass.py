import numpy as np

class NeuralNetwork:
    def __init__(self, network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der):
        self.network_input_size =  network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.activation_ders = activation_ders
        self.cost_fun = cost_fun
        self.cost_der = cost_der

        self.layers = self._create_layers(network_input_size, layer_output_sizes)
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers]
        self.method = "PLAIN"
        self.gamma = 0.001
        self.t = 0
        self.rng = np.random.default_rng(2026)

    def _init_state(self):
        self.vrm = [(np.zeros((3, *W.shape)), np.zeros((3, *b.shape))) for W, b in self.layers]
        self.t = 0

        return self.vrm, self.t


    def _create_layers(self, network_input_size, layer_output_sizes):
        layers = []

        layer_input_size = network_input_size
        for layer_output_size in layer_output_sizes:
            W = self.rng.standard_normal((layer_output_size, layer_input_size))
            b = self.rng.standard_normal((layer_output_size, 1))
            layers.append((W, b))

            layer_input_size = layer_output_size

        return layers

    def _feed_forward(self, inputs):
        layer_inputs = []
        layer_outputs = []
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            layer_inputs.append(a)

            z = W @ a + b
            a = activation_func(z)

            layer_outputs.append(z)

        return layer_inputs, layer_outputs, a

    def _adaptive_method_step(self, method, theta, grad, vrm, gamma, t, beta=0.9, beta1=0.9, beta2=0.999, rho=0.99, eps=1e-8):

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

            return theta - (gamma*m_hat)/(np.sqrt(r_hat + eps))
        
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

    def predict(self, inputs):
        a = inputs

        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            z = W @ a + b
            a = activation_func(z)

        return a

    def cost(self, inputs, targets):
        final_output = self.predict(inputs)
        
        return self.cost_fun(final_output, targets)

    def compute_gradient(self, inputs, targets):
        layer_inputs, layer_outputs, final_output = self._feed_forward(inputs)
        layer_grads = [() for _ in self.layers]

        for i in reversed(range(len(self.layers))):
            a, z, activation_der = layer_inputs[i], layer_outputs[i], self.activation_ders[i]

            if i == len(self.layers) - 1:
                dC_da  = self.cost_der(final_output, targets)

            else:
                (W, b) = self.layers[i + 1]
                dC_da = W.T @ dC_dz

            dC_dz = dC_da*activation_der(z)
            dC_db = jnp.sum(dC_dz, axis=1, keepdims=True)
            dC_dW = dC_dz @ a.T

            layer_grads[i] = (dC_dW, dC_db)

        return layer_grads

    def update_weights(self, layer_grads):

        self.layers = [(self._adaptive_method_step(self.method, W, W_g, vrm_W, self.gamma, self.t),
                        self._adaptive_method_step(self.method, b, b_g, vrm_b, self.gamma, self.t))
                        for (W, b), (W_g, b_g), (vrm_W, vrm_b) in zip(self.layers, layer_grads, self.vrm)]

        return self.layers

    def train(self, inputs, targets, epochs=100, M=None):
        if M is None:
            M = inputs.shape[1]

        self._init_state()
        costs = []

        for epoch in range(1, epochs + 1):
            for idx in self._get_minibatches(inputs.shape[1], M):
                inputs_batch, targets_batch = inputs[:, idx], targets[:, idx]

                self.t += 1
                grads = self.compute_gradient(inputs_batch, targets_batch)
                self.layers = self.update_weights(grads)

            costs.append(self.cost(inputs, targets))

        return costs

    # These last two methods are not needed in the project, but they can be nice to have!
    # The first one has a layers parameter so that you can use jax.grad on it.
    def jax_compliant_predict(self, layers, inputs):
        pass

    def jax_gradient(self, inputs, targets):
        pass