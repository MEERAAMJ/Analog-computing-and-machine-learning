#!/usr/bin/env python3
#
# Fit function to data

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad, jit, vmap, random

# Generate some noisy data

key = random.PRNGKey(54321)# JAX RNG

data_gen = jit(lambda x: jnp.exp(-4.0*x) * jnp.sin(2*jnp.pi*10*x))

N = 200
sigma = 0.05
x = jnp.linspace(0,1,N)
y = vmap(data_gen)(x) + sigma*random.normal(key,shape=(N,))

#plt.plot(x,y)
#plt.show()

# Match function to data

def func(params, x):
  # Parameterised damped oscillation
  l, omega = params
  # Note, we "normalise" parameters
  y_pred = jnp.exp(l*10 * x) * jnp.sin(2*jnp.pi* omega*10 * x)
  return y_pred

def loss(params, x, y):
  # Loss function
  y_pred = func(params, x)
  return jnp.mean((y - y_pred)**2)

# Compile loss and gradient
c_loss = jit(loss)
d_loss = jit(grad(loss))

# One iteration of gradient descent
def update_params(params, x, y):
  grads = d_loss(params, x, y)
  params = [param - 0.1 * grad for param, grad in zip (params, grads)]
  return params

# Initialise parameters
key = random.PRNGKey(0)
params = [random.normal(key, (1,)), random.normal(key, (1,))]

err = []
for epoch in range(100000):
  err.append(c_loss(params, x, y))
  params = update_params(params, x, y)
err.append(c_loss(params, x, y))

print("Damping:  ", params[0]*10)
print("Frequency:", params[1]*10)

y_pred = func(params, x)

# Plot loss and predictions                                                           
f, ax = plt.subplots(1,2)
ax[0].semilogy(err)
ax[0].set_title("History")
ax[1].plot(x, y, label="ground truth")
ax[1].plot(x, y_pred, label="predictions")
ax[1].legend()
plt.show()
