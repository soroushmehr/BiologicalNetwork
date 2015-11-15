from mnist_model import *

import numpy as np
from sys import stdout
import time

path = "mnist_relaxation_analysis.save"
batch_size = 50000
n_hidden = 500

n_iterations = 10
n_test_values = 100

net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)


net.outside_world.set_train(index=0)

best_eps = []
for k in range(n_iterations):
    energy_list=dict()
    for eps in np.linspace(0.,1.,n_test_values):
        net.initialize_train(index=0)
        for epsilon in best_eps:
            net.relax(lambda_y = 0., epsilon_h = epsilon, epsilon_y = epsilon)
        net.relax(lambda_y = 0., epsilon_h = eps, epsilon_y = eps)
        [energy, norm_grad_hy, _, _, _] = net.relax(lambda_y = 0., epsilon_h = 0., epsilon_y = 0.)
        energy_list[energy.mean()] = eps
    print(energy_list)
    min_energy = min(energy_list.iterkeys())
    epsilon = energy_list[min_energy]
    best_eps.append(epsilon)
print(best_eps)