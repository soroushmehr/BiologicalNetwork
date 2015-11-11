from mnist_model import *

import numpy as np
from sys import stdout
import time

path = "params.save"
batch_size = 20
n_epochs = 500


# parameters for the x-clamped relaxation phase
n_iterations = 100 # maximum number of iterations
threshold = .1 # threshold for the norm of grad_hy E to decide when we have reached a fixed point

# parameters for the learning phase
eps_h  = np.float32(.5)
eps_y  = np.float32(.5)
eps_W1 = np.float32(.2)
eps_W2 = np.float32(.02)



net = Network(path=path, batch_size=batch_size)

n_batches_train = net.train_set_x.shape[0] / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_errors, train_cost, train_energy = 0., 0., 0.
    gW11, gW21, gW12, gW22 = 0., 0., 0., 0.
    # train_energy = energy of the stable configuration (= fixed point) at the end of the x-clamped relaxation phase
    for index in range(n_batches_train):
        net.clamp(index=index)

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = np.float32(2. / (2.+k)) # common value for eps_h and eps_y
            [energy, norm_grad_hy, _, error, mse, _, _] = net.iterate(lambda_x = 1., lambda_y = 0., epsilon_x = 0., epsilon_h = eps, epsilon_y = eps, epsilon_W1 = 0., epsilon_W2 = 0.)
            if norm_grad_hy < threshold or k == n_iterations-1:
                train_errors, train_cost, train_energy = train_errors+error, train_cost+mse, train_energy+energy
                error_rate = 100. * train_errors / (index+1)
                cost = train_cost / (index+1)
                energy = train_energy / (index+1)
                break

        # LEARNING PHASE
        [_, _, _, _, _, norm_grad_W1_1, norm_grad_W2_1] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, epsilon_W1 = 0., epsilon_W2 = eps_W2)
        [_, _, _, _, _, norm_grad_W1_2, norm_grad_W2_2] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, epsilon_W1 = eps_W1, epsilon_W2 = 0.)
        gW11, gW21, gW12, gW22 = gW11+norm_grad_W1_1, gW21+norm_grad_W2_1, gW12+norm_grad_W1_2, gW22+norm_grad_W2_2
        g11, g21, g12, g22 = gW11 / (index+1), gW21 / (index+1), gW12 / (index+1), gW22 / (index+1)
        stdout.write("\r %02d E=%.1f er=%.2f%% MSE=%.4f gW11=%.4f gW21=%.2f gW12=%.4f gW22=%.2f" % (epoch, energy, error_rate, cost, g11, g21, g12, g22))
        stdout.flush()

    stdout.write("\n")
    net.save()