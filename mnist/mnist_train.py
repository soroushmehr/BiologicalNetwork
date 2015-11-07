from mnist_model import *

import numpy as np
from sys import stdout
import time

path = "params.save"
batch_size = 1000
n_epochs = 100
n_inference_steps = 10
#n_learning_steps = 2

# parameters for the inference phase
eps_inference = .1 # value of epsilon_x, epsilon_h and epsilon_y

# parameters for the learning phase
eps_h = .1
eps_y = .5
eps_W1 = .1
eps_W2 = .1



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

        # INFERENCE
        for k in range(n_inference_steps):
            [energy, prediction, error, loss] = net.iterative_step(lambda_x = 1., lambda_y = 0., epsilon_x = eps_inference, epsilon_h = eps_inference, epsilon_y = eps_inference, epsilon_W1 = 0., epsilon_W2 = 0.)
            mean_energy = np.mean(energy)
            error_rate = np.mean(train_errors+[error])
            loss_rate = np.mean(train_loss+[loss])
            duration = (time.clock() - start_time) / 60.
            stdout.write("\r %i-%i-%i, energy = %f, error = %f, loss = %f, %f mn" % (epoch, index, k, mean_energy, error_rate, loss_rate, duration))
            stdout.flush()
            if k==n_inference_steps-1:
                train_errors.append(error)
                train_loss.append(loss)

        # LEARNING
        #for k in range(n_learning_steps):
        net.iterative_step(lambda_x = 1., lambda_y = 1., epsilon_x = .1, epsilon_h = eps_h, epsilon_y = eps_y, epsilon_W1 = 0., epsilon_W2 = eps_W2)
        net.iterative_step(lambda_x = 1., lambda_y = 1., epsilon_x = .1, epsilon_h = eps_h, epsilon_y = eps_y, epsilon_W1 = eps_W1, epsilon_W2 = 0.)

    stdout.write("\n")
    net.save()