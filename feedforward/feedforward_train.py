from feedforward_model import *

import numpy as np
from sys import stdout
import time

path = "params.save"
batch_size = 1000
n_epochs = 100
lr = 0.01

net = Network(path=path, batch_size=batch_size)

n_batches_train = net.train_set_x.shape[0] / batch_size

print("path = %s, batch_size = %i" % (path, batch_size))
start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_errors = []
    train_loss = []
    for index in range(n_batches_train):

        net.clamp(index=index)
        [prediction, error, loss] = net.iterative_step(lr=lr)
        train_errors.append(error)
        train_loss.append(loss)

        error_rate = np.mean(train_errors)
        loss_rate = np.mean(train_loss)
        duration = (time.clock() - start_time) / 60.
        stdout.write("\r %i-%i, error = %f, loss = %f, %f mn" % (epoch, index, error_rate, loss_rate, duration))
        stdout.flush()

    stdout.write("\n")
    net.save()