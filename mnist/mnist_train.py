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
alpha_W1 = np.float32(.2)
alpha_W2 = np.float32(.02)



net = Network(path=path, batch_size=batch_size)

n_batches_train = net.train_set_x.get_value(borrow=True).shape[0] / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_energy, train_error, train_cost = 0., 0., 0.
    relax_iterations, relax_fail = 0., 0.
    gW11, gW21, gW12, gW22 = 0., 0., 0., 0.
    # train_energy = energy of the stable configuration (= fixed point) at the end of the x-clamped relaxation phase
    for index in xrange(n_batches_train):
        net.initialize(index=index)

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = np.float32(2. / (2.+k)) # common value for eps_h and eps_y
            [energy, norm_grad_hy, _, error, cost] = net.relax(lambda_y = 0., epsilon_h = eps, epsilon_y = eps)
            if norm_grad_hy < threshold or k == n_iterations-1:
                train_energy, train_error, train_cost = train_energy+energy, train_error+error, train_cost+cost
                relax_iterations, relax_fail = relax_iterations+(k+1.), relax_fail+(k == n_iterations-1)
                energy_avg = train_energy / (index+1)
                error_avg = 100. * train_error / (index+1)
                cost_avg = train_cost / (index+1)
                iterations_avg = relax_iterations / (index+1)
                fail_avg = 100. * relax_fail / (index+1)
                break

        # LEARNING PHASE
        [_, _, _, _, _, Delta_W1_relative_1, Delta_W2_relative_1] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, alpha_W1 = 0., alpha_W2 = alpha_W2)
        [_, _, _, _, _, Delta_W1_relative_2, Delta_W2_relative_2] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = 0.)
        gW11, gW21, gW12, gW22 = gW11+Delta_W1_relative_1, gW21+Delta_W2_relative_1, gW12+Delta_W1_relative_2, gW22+Delta_W2_relative_2
        duration = (time.clock() - start_time) / 60.
        stdout.write("\r %i-%i E=%.1f er=%.2f%% MSE=%.4f it=%.1f fl=%.2f dur=%.1f min" % (epoch, index, energy_avg, error_avg, cost_avg, iterations_avg, fail_avg, duration))
        stdout.flush()

    stdout.write("\n")
    g11, g21, g12, g22 = gW11 / n_batches_train, gW21 / n_batches_train, gW12 / n_batches_train, gW22 / n_batches_train
    stdout.write("gW11=%.4f gW21=%.2f gW12=%.4f gW22=%.2f" % (g11, g21, g12, g22))
    stdout.write("\n")
    
    net.save()