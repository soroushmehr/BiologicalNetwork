from mnist2_model import *

import numpy as np
import sys
from sys import stdout
import time

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "params.save"
batch_size = 20
n_hidden = [500,500]
n_epochs = 500
valid_on = False # validation phase


# hyper-parameters for the x-clamped relaxation phase
n_iterations = 100 # maximum number of iterations
threshold = .1 # threshold for the norm of grad_hy E to decide when we have reached a fixed point

# hyper-parameters for the learning phase
eps_h1   = np.float32(.5)
eps_h2   = np.float32(.5)
eps_y    = np.float32(.5)
alpha_W1 = np.float32(.25)
alpha_W2 = np.float32(.05)
alpha_W3 = np.float32(.003)

net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)

n_batches_train = net.outside_world.train_set_size / batch_size
n_batches_valid = net.outside_world.valid_set_size / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_energy, train_error, train_cost = 0., 0., 0.
    relax_iterations, relax_fail = 0., 0.
    gW1f1, gW2f1, gW3f1, gW1b1, gW2b1, gW3b1 = 0., 0., 0., 0., 0., 0.
    gW1f2, gW2f2, gW3f2, gW1b2, gW2b2, gW3b2 = 0., 0., 0., 0., 0., 0.
    gW1f3, gW2f3, gW3f3, gW1b3, gW2b3, gW3b3 = 0., 0., 0., 0., 0., 0.
    # train_energy = energy of the stable configuration (= fixed point) at the end of the x-clamped relaxation phase
    for index in xrange(n_batches_train):
        net.outside_world.set(index_new=index, dataset_new=1) # dataset_new=1 means training set
        net.initialize()

        # X-CLAMPED RELAXATION PHASE
        for k in range(n_iterations):
            eps = np.float32(10. / (10.+k)) # common value for eps_h and eps_y
            [energy, norm_grad_hy, _, error, cost] = net.relax(epsilon_h1 = eps, epsilon_h2 = eps, epsilon_y = eps)
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
        [_, _, _, _, _, Delta_logW1_fwd_1, Delta_logW1_bwd_1, Delta_logW2_fwd_1, Delta_logW2_bwd_1, Delta_logW3_fwd_1, Delta_logW3_bwd_1] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = 0., alpha_W2 = 0., alpha_W3 = alpha_W3)
        [_, _, _, _, _, Delta_logW1_fwd_2, Delta_logW1_bwd_2, Delta_logW2_fwd_2, Delta_logW2_bwd_2, Delta_logW3_fwd_2, Delta_logW3_bwd_2] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = 0., alpha_W2 = alpha_W2, alpha_W3 = 0.)
        [_, _, _, _, _, Delta_logW1_fwd_3, Delta_logW1_bwd_3, Delta_logW2_fwd_3, Delta_logW2_bwd_3, Delta_logW3_fwd_3, Delta_logW3_bwd_3] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h1 = eps_h1, epsilon_h2 = eps_h2, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = 0., alpha_W3 = 0.)
        gW1f1, gW2f1, gW3f1, gW1b1, gW2b1, gW3b1 = gW1f1 + Delta_logW1_fwd_1, gW2f1 + Delta_logW2_fwd_1, gW3f1 + Delta_logW3_fwd_1, gW1b1 + Delta_logW1_bwd_1, gW2b1 + Delta_logW2_bwd_1, gW3b1 + Delta_logW3_bwd_1
        gW1f2, gW2f2, gW3f2, gW1b2, gW2b2, gW3b2 = gW1f2 + Delta_logW1_fwd_2, gW2f2 + Delta_logW2_fwd_2, gW3f2 + Delta_logW3_fwd_2, gW1b2 + Delta_logW1_bwd_2, gW2b2 + Delta_logW2_bwd_2, gW3b2 + Delta_logW3_bwd_2
        gW1f3, gW2f3, gW3f3, gW1b3, gW2b3, gW3b3 = gW1f3 + Delta_logW1_fwd_3, gW2f3 + Delta_logW2_fwd_3, gW3f3 + Delta_logW3_fwd_3, gW1b3 + Delta_logW1_bwd_3, gW2b3 + Delta_logW2_bwd_3, gW3b3 + Delta_logW3_bwd_3
        stdout.write("\r%i-train-%i E=%.1f er=%.2f%% MSE=%.4f it=%.1f fl=%.1f%%" % (epoch, index, energy_avg, error_avg, cost_avg, iterations_avg, fail_avg))
        stdout.flush()

    stdout.write("\n")
    dlogW1f1, dlogW2f1, dlogW3f1, dlogW1b1, dlogW2b1, dlogW3b1 = 100. * gW1f1 / n_batches_train, 100. * gW2f1 / n_batches_train, 100. * gW3f1 / n_batches_train, 100. * gW1b1 / n_batches_train, 100. * gW2b1 / n_batches_train, 100. * gW3b1 / n_batches_train
    dlogW1f2, dlogW2f2, dlogW3f2, dlogW1b2, dlogW2b2, dlogW3b2 = 100. * gW1f2 / n_batches_train, 100. * gW2f2 / n_batches_train, 100. * gW3f2 / n_batches_train, 100. * gW1b2 / n_batches_train, 100. * gW2b2 / n_batches_train, 100. * gW3b2 / n_batches_train
    dlogW1f3, dlogW2f3, dlogW3f3, dlogW1b3, dlogW2b3, dlogW3b3 = 100. * gW1f3 / n_batches_train, 100. * gW2f2 / n_batches_train, 100. * gW3f3 / n_batches_train, 100. * gW1b3 / n_batches_train, 100. * gW2b3 / n_batches_train, 100. * gW3b3 / n_batches_train
    print("   k=1 backward: dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1b1, dlogW2b1, dlogW3b1))
    print("   k=1 forward:  dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1f1, dlogW2f1, dlogW3f1))
    print("   k=2 backward: dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1b2, dlogW2b2, dlogW3b2))
    print("   k=2 forward:  dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1f2, dlogW2f2, dlogW3f2))
    print("   k=3 backward: dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1b3, dlogW2b3, dlogW3b3))
    print("   k=3 forward:  dlogW1=%.3f%% dlogW2=%.3f%% dlogW3=%.3f%%" % (dlogW1f3, dlogW2f3, dlogW3f3))

    # VALIDATION
    if valid_on:
        valid_energy, valid_error, valid_cost = 0., 0., 0.
        relax_iterations, relax_fail = 0., 0.

        for index in xrange(n_batches_valid):
            net.outside_world.set(index_new=index, dataset_new=2) # dataset_new=2 means validation set
            net.initialize()

            # X-CLAMPED RELAXATION PHASE
            for k in range(n_iterations):
                eps = np.float32(10. / (10.+k)) # common value for eps_h and eps_y
                [energy, norm_grad_hy, _, error, cost] = net.relax(epsilon_h1 = eps, epsilon_h2 = eps, epsilon_y = eps)
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
        stdout.write("\n")

    duration = (time.clock() - start_time) / 60.
    print("   dur=%.1f min" % (duration))
    net.save_params()