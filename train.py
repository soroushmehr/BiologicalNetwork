from biologicalnetwork import *

import numpy as np
from sys import stdout
import time

batch_size = 500
n_epochs = 100
n_inference_steps = 10
n_learning_steps = 2

# parameters for the learning phase
lambda_x = 1.
lambda_y = 1.
eps_s = 0.1
eps_w = 0.1


net = Network(batch_size=batch_size)
inference_step = net.build_inference_step()

train_set, valid_set, test_set = mnist()
train_set_x, train_set_y = train_set
valid_set_x, valid_set_y = valid_set
test_set_x, test_set_y = test_set

n_batches_train = train_set_x.shape[0] / batch_size
n_batches_valid = valid_set_x.shape[0] / batch_size

start_time = time.clock()
for e in range(n_epochs):

    # TRAINING
    train_errors = []
    for i in range(n_batches_train):
        net.clear(x_value=train_set_x[i*batch_size:(i+1)*batch_size,])
        for k in range(n_inference_steps):
            [energy, prediction, error] = inference_step(train_set_x[i*batch_size:(i+1)*batch_size,], train_set_y[i*batch_size:(i+1)*batch_size,], lambda_x, 0., eps_s, 0.)
            error_rate = np.mean(train_errors+[error])
            timestamp = (time.clock() - start_time) / 60.
            stdout.write("\repoch %i, training: batch %i, step %i, error rate %f, duration %f min" % (e, i, k, error_rate, timestamp))
            stdout.flush()
            if k==n_inference_steps-1:
                train_errors.append(error)
        for k in range(n_learning_steps):
            inference_step(train_set_x[i*batch_size:(i+1)*batch_size,], train_set_y[i*batch_size:(i+1)*batch_size,], lambda_x, lambda_y, eps_s, eps_w)
    stdout.write("\n")

    # VALIDATION
    valid_errors = []
    for i in range(n_batches_valid):
        net.clear(x_value=valid_set_x[i*batch_size:(i+1)*batch_size,])
        for k in range(n_inference_steps):
            [energy, prediction, error] = inference_step(valid_set_x[i*batch_size:(i+1)*batch_size,], valid_set_y[i*batch_size:(i+1)*batch_size,], lambda_x, 0., eps_s, 0.)
            error_rate = np.mean(valid_errors+[error])
            timestamp = (time.clock() - start_time) / 60.
            stdout.write("\repoch %i, validation: batch %i, step %i, error rate %f, duration %f min" % (e, i, k, error_rate, timestamp))
            stdout.flush()
            if k==n_inference_steps-1:
                valid_errors.append(error)
    stdout.write("\n")
    net.save()