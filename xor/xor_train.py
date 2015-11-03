from xor_model import *

import numpy as np
from sys import stdout
import time

n_epochs = 1000
n_inference_steps = 100
n_learning_steps = 2

# parameters for the learning phase
lambda_x = 1.
lambda_y = 1.
eps_s = 0.01
eps_w = 0.001
eps_y = 0.2


net = Network(batch_size=4)
inference_step = net.build_inference_step()

start_time = time.clock()
for epoch in range(n_epochs):
    net.clamp(index=0)
    for k in range(n_inference_steps):
        [energy, prediction, error, loss] = inference_step(lambda_x, 0., eps_s, 0., 0.)
        duration = (time.clock() - start_time) / 60.
        stdout.write("\rep %i, step %i, error %f, loss %f, dur %f min" % (epoch, k, error, loss, duration))
        stdout.flush()
    for k in range(n_learning_steps):
        inference_step(lambda_x, lambda_y, eps_s, eps_w, eps_y)
    stdout.write("\n")
net.save()