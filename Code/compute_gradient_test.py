"""Test file for the compute_gradient method in NeuralNetworkClass class. The testing is done by comparing the method against results
from the package JAX and seeing if they produce similar enough results"""

from Functions import identity, identity_der, sigmoid, sigmoid_der, mse, mse_der, runge
from Functions_JAX import identity_JAX, sigmoid_JAX, mse_JAX
from NeuralNetworkClass import NeuralNetwork
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  

#  --- Parameters --- #
lam = 1e-3 
lr = 0.01
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

# --- JAX versions of the functions --- #
jax_activation_funcs = [sigmoid_JAX, sigmoid_JAX, identity_JAX]

# --- Compare gradients --- #
for penalty in (None, "L1", "L2"):
    NN = NeuralNetwork(network_input_size, layer_output_sizes, activation_funcs, activation_ders, cost_fun, cost_der)
    NN.change_seed(seed)
    NN._init_state()  
    NN.learning_rate(lr)
    NN.add_penalty(lam, penalty)

    grads_own = NN.compute_gradient(x, y)
    grads_jax = NN.compute_gradient_JAX(x, y, activation_funcs=jax_activation_funcs, cost_fun=mse_JAX)

    print(f"Penalty: {penalty}")
    for i, ((W_own, b_own), (W_jax, b_jax)) in enumerate(zip(grads_own, grads_jax)):
        W_err = np.max(np.abs(W_own - W_jax))
        b_err = np.max(np.abs(b_own - b_jax))
        print(f"  Layer {i}: max |dW diff| = {W_err:.2e}, max |db diff| = {b_err:.2e}, "
              f"match = {np.allclose(W_own, W_jax) and np.allclose(b_own, b_jax)}")
        
    print("")