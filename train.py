from biologicalnetwork import *

import numpy as np
from sys import stdout
import time

batch_size = 1000
n_epochs = 100
n_inference_steps = 10
#n_learning_steps = 2

# parameters for the learning phase
lambda_x = 1.
lambda_y = 1.
eps_s = 0.1
eps_W1 = 0.1
eps_W2 = 0.1
eps_y = 0.5


net = Network(batch_size=batch_size)
inference_step = net.build_inference_step()

n_batches_train = net.train_set_x.shape[0] / batch_size

start_time = time.clock()
for epoch in range(n_epochs):

    # TRAINING
    train_errors = []
    train_loss = []
    for index in range(n_batches_train):
        net.clamp(index=index)
        for k in range(n_inference_steps):
            [energy, prediction, error, loss] = inference_step(lambda_x, 0., eps_s, 0., 0., 0.)
            error_rate = np.mean(train_errors+[error])
            loss_rate = np.mean(train_loss+[loss])
            duration = (time.clock() - start_time) / 60.
            stdout.write("\rep %i, ba %i, %i, error %f, loss %f, dur %f min" % (epoch, index, k, error_rate, loss_rate, duration))
            stdout.flush()
            if k==n_inference_steps-1:
                train_errors.append(error)
                train_loss.append(loss)
        #for k in range(n_learning_steps):
        inference_step(lambda_x, lambda_y, eps_s, 0., eps_W2, eps_y)
        inference_step(lambda_x, lambda_y, eps_s, eps_W1, 0., eps_y)
    stdout.write("\n")
    net.save()