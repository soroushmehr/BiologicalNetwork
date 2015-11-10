from unsupervised_model import *

import numpy as np
from sys import stdout
import time

path = "params.save"
batch_size = 100
n_epochs = 500


# parameters for the x-clamped relaxation phase
n_iterations = 30 # 

# parameters for the learning phase
eps_x  = .5
eps_h  = .1
eps_W1 = .1



net = Network(path=path, batch_size=batch_size)

n_batches_train = net.train_set_x.shape[0] / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_cost = []
    for index in range(n_batches_train):
        net.clamp(index=index)

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = 2. / (2.+k) # common value for eps_h and eps_y
            [energy, norm_grad_x, norm_grad_h, mse] = net.iterate(lambda_x = 1., epsilon_x = 0., epsilon_h = eps, epsilon_W1 = 0.)
            mean_energy = np.mean(energy)
            stdout.write("\r %i-%i-%i, E = %.1f, norm = %.1f" % (epoch, index, k, mean_energy, norm_grad_h))
            stdout.flush()
            if norm_grad_h < 0.1 or k == n_iterations-1:
                break

        # FREE RELAXATION PHASE
        for k in range(n_iterations):
            eps = 2. / (2.+k) # common value for eps_h and eps_y
            [energy, norm_grad, prediction, error, mse] = net.iterate(lambda_x = 0., epsilon_x = eps, epsilon_h = eps, epsilon_W1 = 0.)
            mean_energy = np.mean(energy)
            cost = np.mean(train_cost+[mse])
            duration = (time.clock() - start_time) / 60.
            stdout.write("\r %i-%i-%i, E = %.1f, norm = %.1f, MSE = %.4f, %.1f min" % (epoch, index, k, mean_energy, norm_grad_x+norm_grad_h, cost, duration))
            stdout.flush()
            if (norm_grad_x+norm_grad_h) < 0.1 or k == n_iterations-1:
                train_cost.append(mse)
                break

        # LEARNING PHASE
        net.iterate(lambda_x = 1., epsilon_x = eps_x, epsilon_h = eps_h, epsilon_W1 = eps_W1)

    stdout.write("\n")
    net.save()