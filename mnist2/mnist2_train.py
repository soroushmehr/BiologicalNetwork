from mnist2_model import *

import numpy as np
from sys import stdout
import time

path = "params.save"
batch_size = 20
n_hidden = [500,500]
n_epochs = 500


# parameters for the x-clamped relaxation phase
n_iterations = 100 # maximum number of iterations
threshold = 1. # threshold for the norm of grad_hy E to decide when we have reached a fixed point

# parameters for the learning phase
eps_h1 = np.float32(.5)
eps_h2 = np.float32(.5)
eps_y  = np.float32(.5)
alpha_W1 = np.float32(.2)
alpha_W2 = np.float32(.008)
alpha_W3 = np.float32(.008)

net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)

n_batches_train = net.outside_world.train_set_x.get_value(borrow=True).shape[0] / batch_size
n_batches_valid = net.outside_world.valid_set_x.get_value(borrow=True).shape[0] / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_energy, train_error, train_cost = 0., 0., 0.
    relax_iterations, relax_fail = 0., 0.
    gW11, gW21, gW31, gW12, gW22, gW32, gW13, gW23, gW33 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # train_energy = energy of the stable configuration (= fixed point) at the end of the x-clamped relaxation phase
    for index in xrange(n_batches_train):
        net.outside_world.set_train(index=index)
        net.initialize_train(index=index)

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = np.float32(2. / (2.+k)) # common value for eps_h and eps_y
            [energy, norm_grad_hy, _, error, cost] = net.relax(lambda_y = 0., epsilon_h1 = eps, epsilon_h2 = eps, epsilon_y = eps)
            #print(energy.mean(), norm_grad_hy.mean())
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
        [_, _, _, _, _, Delta_W1_relative_1, Delta_W2_relative_1, Delta_W3_relative_1] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = alpha_W2, alpha_W3 = alpha_W3)
        [_, _, _, _, _, Delta_W1_relative_2, Delta_W2_relative_2, Delta_W3_relative_2] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = alpha_W2, alpha_W3 = alpha_W3)
        [_, _, _, _, _, Delta_W1_relative_3, Delta_W2_relative_3, Delta_W3_relative_3] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = alpha_W2, alpha_W3 = alpha_W3)
        gW11, gW21, gW31 = gW11+Delta_W1_relative_1, gW21+Delta_W2_relative_1, gW31+Delta_W3_relative_1
        gW12, gW22, gW32 = gW12+Delta_W1_relative_2, gW22+Delta_W2_relative_2, gW32+Delta_W3_relative_2
        gW13, gW23, gW33 = gW13+Delta_W1_relative_3, gW23+Delta_W2_relative_3, gW33+Delta_W3_relative_3
        stdout.write("\r%i-train-%i E=%.1f er=%.2f%% MSE=%.4f it=%.1f fl=%.1f%%" % (epoch, index, energy_avg, error_avg, cost_avg, iterations_avg, fail_avg))
        stdout.flush()

    stdout.write("\n")
    g11, g21, g31 = 100. * gW11 / n_batches_train, 100. * gW21 / n_batches_train, 100. * gW31 / n_batches_train
    g12, g22, g32 = 100. * gW12 / n_batches_train, 100. * gW22 / n_batches_train, 100. * gW32 / n_batches_train
    g13, g23, g33 = 100. * gW13 / n_batches_train, 100. * gW23 / n_batches_train, 100. * gW33 / n_batches_train
    duration = (time.clock() - start_time) / 60.
    stdout.write("   gW11=%.3f%% gW12=%.3f%% gW13=%.3f%% dur=%.1f min\n" % (g11, g12, g13, duration))
    stdout.write("   gW21=%.3f%% gW22=%.3f%% gW23=%.3f%%\n" % (g21, g22, g23))
    stdout.write("   gW31=%.3f%% gW32=%.3f%% gW33=%.3f%%\n" % (g31, g32, g33))

    # VALIDATION
    '''valid_energy, valid_error, valid_cost = 0., 0., 0.
    relax_iterations, relax_fail = 0., 0.

    for index in xrange(n_batches_valid):
        net.outside_world.set_valid(index=index)
        net.initialize_valid(index=index)

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = np.float32(2. / (2.+k)) # common value for eps_h and eps_y
            [energy, norm_grad_hy, _, error, cost] = net.relax(lambda_y = 0., epsilon_h = eps, epsilon_y = eps)
            if norm_grad_hy < threshold or k == n_iterations-1:
                valid_energy, valid_error, valid_cost = valid_energy+energy, valid_error+error, valid_cost+cost
                relax_iterations, relax_fail = relax_iterations+(k+1.), relax_fail+(k == n_iterations-1)
                energy_avg = valid_energy / (index+1)
                error_avg = 100. * valid_error / (index+1)
                cost_avg = valid_cost / (index+1)
                iterations_avg = relax_iterations / (index+1)
                fail_avg = 100. * relax_fail / (index+1)
                break

        stdout.write("\r   valid-%i  E=%.1f er=%.2f%% MSE=%.4f it=%.1f fl=%.1f%%" % (index, energy_avg, error_avg, cost_avg, iterations_avg, fail_avg))
        stdout.flush()
    stdout.write("\n")'''
    
    net.save_params()