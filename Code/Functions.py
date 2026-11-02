import jax.numpy as jnp

def ReLU(z):
    return jnp.where(z > 0, z, 0.0)

def ReLU_der(z):
    return jnp.where(z > 0, 1.0, 0.0)

def leaky_ReLU(z, a=0.01):
    return jnp.where(z > 0, z, a*z)

def leaky_ReLU_der(z, a=0.01):
    return jnp.where(z > 0, 1, a)

def sigmoid(z):
    return 1/(1 + jnp.exp(-z))

def sigmoid_der(z):
    sig = sigmoid(z)
    
    return sig*(1 - sig)

def cross_entropy(predict, target):
    return jnp.mean(jnp.sum(-target*jnp.log(predict), axis=1))

def softmax(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z/jnp.sum(e_z, axis=1, keepdims=True)

def mse(predict, target):
    return jnp.mean((predict - target)**2)

def mse_der(predict, target):
    return (2/predict.size)*(predict - target)

def runge(x):
    return 1/(1 + 25*x**2)