from mlp_model import *

import numpy as np
import sys
from sys import stdout
import time

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = "params.save"
batch_size = 20
n_hidden = 500
n_epochs = 500
valid_on = True # validation phase
alpha = np.float32(.05)

net = Network(path=path, batch_size=batch_size, n_hidden=n_hidden)

n_batches_train = net.outside_world.train_set_size / batch_size
n_batches_valid = net.outside_world.valid_set_size / batch_size

print("MLP, path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_error, train_cost = 0., 0.
    for index in xrange(n_batches_train):
        net.outside_world.set(index_new=index, dataset_new=1) # dataset_new=1 means training set
        [error, cost] = net.train(alpha)
        train_error, train_cost = train_error+error, train_cost+cost
        error_avg = 100. * train_error / (index+1)
        cost_avg = train_cost / (index+1)
        stdout.write("\r%2i-train-%3i er=%.2f%% MSE=%.4f" % (epoch, index, error_avg, cost_avg))
        stdout.flush()

    stdout.write("\n")

    # VALIDATION
    if valid_on:
        valid_error, valid_cost = 0., 0.

        for index in xrange(n_batches_valid):
            net.outside_world.set(index_new=index, dataset_new=2) # dataset_new=2 means validation set
            [error, cost] = net.predict()
            valid_error, valid_cost = valid_error+error, valid_cost+cost
            error_avg = 100. * valid_error / (index+1)
            cost_avg = valid_cost / (index+1)

            stdout.write("\r   valid-%i  er=%.2f%% MSE=%.4f" % (index, error_avg, cost_avg))
            stdout.flush()
        stdout.write("\n")

    duration = (time.clock() - start_time) / 60.
    print("   dur=%.1f min" % (duration))
    net.save_params()