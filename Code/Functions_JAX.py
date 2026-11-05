"""File similiar to Functions.py but numpy being replaced by JAX numpy"""


import jax.numpy as jnp


def identity_JAX(z):
    return z

def ReLU_JAX(z):
    return jnp.where(z > 0, z, 0.0)


def leaky_ReLU_JAX(z, a=0.01):
    return jnp.where(z > 0, z, a*z)

def sigmoid_JAX(z):
    return 1/(1 + jnp.exp(-z))

def mse_JAX(predict, target):
    return jnp.mean((predict - target)**2)

def softmax_JAX(z):
    e_z = jnp.exp(z - jnp.max(z, axis=1, keepdims=True))
    return e_z/jnp.sum(e_z, axis=1, keepdims=True)

def cross_entropy_JAX(predict, target):
    return jnp.mean(jnp.sum(-target*jnp.log(predict), axis=1))
