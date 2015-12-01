import cPickle
import numpy as np
import os
import sys
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

path = os.path.dirname(os.path.abspath(__file__))+os.sep+os.pardir
sys.path.insert(0, path)
from outside_world import Outside_World

def initialize_layer(n_in, n_out):

    rng = np.random.RandomState()

    W_values = np.asarray(
        rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        ),
        dtype=theano.config.floatX
    )

    return W_values

def load(path):
    f = file(path, 'rb')
    params = cPickle.load(f)
    f.close()
    return params

def mnist():
    f = gzip.open("mnist.pkl.gz", 'rb')
    datasets = cPickle.load(f)
    f.close()
    return datasets

def rho(x):
    # return T.clip(x, 0., 1.)   # hard-sigmoid
    # return T.nnet.sigmoid(x-2) # sigmoid
    return T.tanh(x)           # hyperbolic tangent

def rho_prime(x):
    # return (x > 0.) * (x < 1.) # hard-sigmoid
    # y = T.nnet.sigmoid(x-2)    # sigmoid
    # return y * (1 - y)         # sigmoid
    y = T.tanh(x)              # hyperbolic tangent
    return 1 - y ** 2          # hyperbolic tangent

class Network(object):

    def __init__(self, path="params.save", batch_size=1):

        self.path = path

        # LOAD/INITIALIZE PARAMETERS
        if not os.path.isfile(self.path):
            bx_values = np.zeros((28*28,), dtype=theano.config.floatX)
            W1_values = initialize_layer(28*28, 500)
            bh_values = np.zeros((500,), dtype=theano.config.floatX)
        else:
            [bx_values, W1_values, bh_values] = load(self.path)

        self.bx = theano.shared(value=bx_values, name='bx', borrow=True)
        self.W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        self.bh = theano.shared(value=bh_values, name='bh', borrow=True)

        self.params = [self.bx, self.W1, self.bh]

        # LOAD DATASETS
        train_set, valid_set, test_set = mnist()
        self.train_set_x, _ = train_set

        # INITIALIZE STATES
        self.batch_size = batch_size
        self.rng = np.random.RandomState()

        self.x_data = theano.shared(value=np.zeros((self.batch_size, 28*28)), name='x_data', borrow=True)
        self.x      = theano.shared(value=np.zeros((self.batch_size, 28*28)), name='x',      borrow=True)
        self.h      = theano.shared(value=np.zeros((self.batch_size, 500)),   name='h',      borrow=True)

        self.clamp(index=0)

        #self.theano_rng = RandomStreams(self.rng.randint(2 ** 30)) # will be used when introducing noise in Langevin MCMC

        self.mse = T.mean(((self.x - self.x_data) ** 2).sum(axis=1))

        self.iterate = self.build_iterative_function()

    def save(self):
        f = file(self.path, 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def clamp(self, index, clear=True):
        self.x_data.set_value(self.train_set_x[index*self.batch_size:(index+1)*self.batch_size,])
        if clear:
            self.x.set_value(self.train_set_x[index*self.batch_size:(index+1)*self.batch_size,])
            self.h.set_value(np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size, 500) ),
                dtype=theano.config.floatX
            ))

    def energy(self):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        squared_norm = ( T.batched_dot(self.x,self.x) + T.batched_dot(self.h,self.h) ) / 2
        uni_terms    = - T.dot(rho_x, self.bx) - T.dot(rho_h, self.bh)
        bi_terms     = - T.batched_dot( T.dot(rho_x, self.W1), rho_h )
        return squared_norm + uni_terms + bi_terms


    def build_iterative_function(self):

        def states_dot(lambda_x):
            rho_x = rho(self.x)
            rho_h = rho(self.h)

            x_pressure = rho_prime(self.x) * (T.dot(rho_h, self.W1.T) + self.bx)
            x_pressure_final = lambda_x * self.x_data + (1 - lambda_x) * x_pressure
            x_dot = x_pressure_final - self.x

            h_pressure = rho_prime(self.h) * (T.dot(rho_x, self.W1) + self.bh)
            h_dot = h_pressure - self.h

            return [x_dot, h_dot]

        def params_delta(x_delta, h_delta):
            rho_x = rho(self.x)
            rho_h = rho(self.h)

            bx_delta = T.mean(x_delta, axis=0)
            W1_delta = (T.dot(x_delta.T, rho_h) + T.dot(rho_x.T, h_delta)) / self.x.shape[0]
            bh_delta = T.mean(h_delta, axis=0)

            return [bx_delta, W1_delta, bh_delta]

        lambda_x = T.dscalar('lambda_x')
        epsilon_x = T.dscalar('epsilon_x')
        epsilon_h = T.dscalar('epsilon_h')
        epsilon_W1 = T.dscalar('epsilon_W1')

        [x_dot, h_dot] = states_dot(lambda_x)

        x_delta = epsilon_x * x_dot
        h_delta = epsilon_h * h_dot

        [bx_delta, W1_delta, bh_delta] = params_delta(x_delta, h_delta)

        x_new = self.x + x_delta
        h_new = self.h + h_delta

        bx_new = self.bx + epsilon_W1 * bx_delta
        W1_new = self.W1 + epsilon_W1 * W1_delta
        bh_new = self.bh + epsilon_W1 * bh_delta

        updates = [(self.x,x_new), (self.h,h_new), (self.bx,bx_new), (self.W1,W1_new), (self.bh,bh_new)]

        norm_grad_x = T.sqrt( (x_dot ** 2).mean(axis=0).sum())
        norm_grad_h = T.sqrt( (h_dot ** 2).mean(axis=0).sum())

        iterative_function = theano.function(
            inputs=[lambda_x, epsilon_x, epsilon_h, epsilon_W1],
            outputs=[self.energy(), norm_grad_x, norm_grad_h, self.mse],
            updates=updates
        )

        return iterative_function