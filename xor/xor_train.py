from xor_model import *

import numpy as np
from sys import stdout
import time

path = "xor_params.save"
batch_size=4
n_hidden = 3
n_epochs = 1000


# parameters for the x-clamped relaxation phase
n_iterations = 100 # maximum number of iterations
threshold = .01 # threshold for the norm of grad_hy E to decide when we have reached a fixed point

# parameters for the learning phase
eps_h  = np.float32(.5)
eps_y  = np.float32(.5)
alpha_W1 = np.float32(.5)
alpha_W2 = np.float32(.1)


net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)
net.outside_world.set(index_new=0)

start_time = time.clock()
print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING

    net.initialize()

    # X-CLAMPED RELAXATION PHASE
    for k in range(n_iterations):
        eps = np.float32(2. / (2. + k)) # common value for eps_h and eps_y
        [energy, norm_grad_hy, _, error, cost] = net.relax(epsilon_h = eps, epsilon_y = eps)
        if norm_grad_hy < threshold or k == n_iterations-1:
            iterations = k+1.
            fail = (k == n_iterations-1)
            break

    # LEARNING PHASE
    [_, _, _, _, _, Delta_logW1_1, Delta_logW2_1] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = alpha_W2)
    [_, _, _, _, _, Delta_logW1_2, Delta_logW2_2] = net.iterate(lambda_x = 1., lambda_y = 1., epsilon_x = 0., epsilon_h = eps_h, epsilon_y = eps_y, alpha_W1 = alpha_W1, alpha_W2 = alpha_W2)
    
    
    g11, g21, g12, g22 = 100. * Delta_logW1_1, 100. * Delta_logW2_1, 100. * Delta_logW1_2, 100. * Delta_logW2_2
    duration = (time.clock() - start_time) / 60.
    print("epoch=%i E=%.1f er=%.2f%% MSE=%.4f it=%.1f fl=%.1f%%" % (epoch, energy, error, cost, iterations, fail))
    print("         gW11=%.3f%% gW12=%.3f%% gW21=%.3f%% gW22=%.3f%% dur=%.1f min" % (g11, g12, g21, g22, duration))
    
net.save_params()