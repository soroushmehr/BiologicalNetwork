from mnist_model import *

import numpy as np
from sys import stdout
import time


# HYPERPARAMETERS
path = "mnist_relaxation_analysis.save"
batch_size = 50000
n_hidden = 500

n_iterations = 10
n_test_values = 101

# INITIALIZE THE NETWORK
net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)
net.outside_world.set_train(index=0)
net.initialize_train(index=0)

# THEANO FUNCTIONS
h_save = theano.shared(value=np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), name='h_save', borrow=True)
y_save = theano.shared(value=np.zeros((batch_size, 10),       dtype=theano.config.floatX), name='y_save', borrow=True)

updates_load = [(net.h, h_save), (net.y, y_save)]
updates_save = [(h_save, net.h), (y_save, net.y)]

save = theano.function(
    inputs=[],
    outputs=[],
    updates=updates_save
)

load = theano.function(
    inputs=[],
    outputs=[],
    updates=updates_load
)




save()
print("Relaxation Analysis: n_iterations=%i n_test_values=%i" % (n_iterations, n_test_values))
start_time = time.clock()
for k in range(n_iterations):
    energy_list=dict()
    for eps in np.linspace(0.,1.,n_test_values):
        load()
        eps = np.float32(eps)
        net.relax(lambda_y = 0., epsilon_h = eps, epsilon_y = eps)
        [energy, norm_grad_hy, _, _, _] = net.relax(lambda_y = 0., epsilon_h = 0., epsilon_y = 0.)
        energy_list[energy.mean()] = eps, norm_grad_hy
        stdout.write("\rk=%i eps=%.2f E=%.1f norm_grad_hy=%.1f" % (k, eps, energy.mean(), norm_grad_hy))
        stdout.flush()

    min_energy = min(energy_list.iterkeys())
    best_epsilon, norm_grad_hy = energy_list[min_energy]
    duration = (time.clock() - start_time) / 60.

    print("k=%i best_epsilon=%.2f E=%.1f norm_grad_hy=%.1f dur=%.1fmin" % (k, best_epsilon, min_energy, norm_grad_hy, duration))

    load()
    net.relax(lambda_y = 0., epsilon_h = best_epsilon, epsilon_y = best_epsilon)
    save()

# EXPERIMENT - RESULTS
# best values of epsilon = [.75, .31, .30, .27, .25, .27, .27, .29, .32, 0.28]