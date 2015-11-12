import cPickle
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.shared_randomstreams import RandomStreams

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
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval

def rho(x):
    # return T.clip(x, 0., 1.)   # hard-sigmoid
    # return T.nnet.sigmoid(x-2) # sigmoid
    return T.tanh(x)           # hyperbolic tangent

def rho_prime(x):
    # return (x > 0.) * (x < 1.) # hard-sigmoid
    # y = T.nnet.sigmoid(x-2)    # sigmoid
    # return y * (1 - y)         # sigmoid
    y = T.tanh(x)              # hyperbolic tangent
    return 1. - y ** 2          # hyperbolic tangent

class Network(object):

    def __init__(self, path="params.save", batch_size=1):

        self.path = path

        # LOAD/INITIALIZE PARAMETERS
        if not os.path.isfile(self.path):
            bx_values = np.zeros((28*28,), dtype=theano.config.floatX)
            W1_values = initialize_layer(28*28, 500)
            bh_values = np.zeros((500,), dtype=theano.config.floatX)
            W2_values = initialize_layer(500, 10)
            by_values = np.zeros((10,), dtype=theano.config.floatX)
        else:
            [bx_values, W1_values, bh_values, W2_values, by_values] = load(self.path)

        self.bx = theano.shared(value=bx_values, name='bx', borrow=True)
        self.W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        self.bh = theano.shared(value=bh_values, name='bh', borrow=True)
        self.W2 = theano.shared(value=W2_values, name='W2', borrow=True)
        self.by = theano.shared(value=by_values, name='by', borrow=True)

        self.params = [self.bx, self.W1, self.bh, self.W2, self.by]

        # LOAD DATASETS
        [(self.train_set_x, self.train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)] = mnist()

        # INITIALIZE STATES
        self.batch_size = batch_size
        

        self.x_data = theano.shared(value=np.zeros((self.batch_size, 28*28), dtype=theano.config.floatX), name='x_data', borrow=True)
        self.x      = theano.shared(value=np.zeros((self.batch_size, 28*28), dtype=theano.config.floatX), name='x',      borrow=True)
        self.h      = theano.shared(value=np.zeros((self.batch_size, 500),   dtype=theano.config.floatX), name='h',      borrow=True)
        self.y      = theano.shared(value=np.zeros((self.batch_size, 10),    dtype=theano.config.floatX), name='y',      borrow=True)
        self.y_data = theano.shared(value=np.zeros((self.batch_size, ),      dtype='int32'),              name='y_data', borrow=True)

        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        rng = np.random.RandomState()
        self.theano_rng = RandomStreams(rng.randint(2 ** 30)) # used to initialize h and y at random at the beginning of the x-clamped relaxation phase. will also be used when introducing noise in Langevin MCMC

        self.prediction = T.argmax(self.y, axis=1)
        self.error_rate = T.mean(T.neq(self.prediction, self.y_data))
        self.mse        = T.mean(((self.y - self.y_data_one_hot) ** 2).sum(axis=1))

        self.initialize, self.clamp = self.build_clamp_function()
        self.iterate, self.relax = self.build_iterative_function()

    def save(self):
        f = file(self.path, 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def build_clamp_function(self):

        index = T.lscalar('index')
        x_data_init = self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
        h_init = self.theano_rng.uniform(size=self.h.shape, low=0., high=1., dtype=theano.config.floatX)
        y_init = self.theano_rng.uniform(size=self.y.shape, low=0., high=1., dtype=theano.config.floatX)
        y_data_init = self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]

        updates_initialize = [(self.x_data, x_data_init), (self.x, x_data_init), (self.h, h_init), (self.y, y_init), (self.y_data, y_data_init)]
        updates_clamp = [(self.x_data, x_data_init), (self.y_data, y_data_init)]

        initialize_function = theano.function(
            inputs=[index],
            outputs=[],
            updates=updates_initialize
        )

        clamp_function = theano.function(
            inputs=[index],
            outputs=[],
            updates=updates_clamp
        )

        return initialize_function, clamp_function


    def energy(self):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        rho_y = rho(self.y)
        squared_norm = ( T.batched_dot(self.x,self.x) + T.batched_dot(self.h,self.h) + T.batched_dot(self.y,self.y) ) / 2.
        uni_terms    = - T.dot(rho_x, self.bx) - T.dot(rho_h, self.bh) - T.dot(rho_y, self.by)
        bi_terms     = - T.batched_dot( T.dot(rho_x, self.W1), rho_h ) - T.batched_dot( T.dot(rho_h, self.W2), rho_y )
        return T.mean( squared_norm + uni_terms + bi_terms )

    def build_iterative_function(self):

        def states_dot(lambda_x, lambda_y):
            rho_x = rho(self.x)
            rho_h = rho(self.h)
            rho_y = rho(self.y)

            R_x = rho_prime(self.x) * (T.dot(rho_h, self.W1.T) + self.bx)
            R_x_total = lambda_x * self.x_data + (1. - lambda_x) * R_x
            x_dot = R_x_total - self.x

            R_h = rho_prime(self.h) * (T.dot(rho_x, self.W1) + T.dot(rho_y, self.W2.T) + self.bh)
            h_dot = R_h - self.h

            R_y = rho_prime(self.y) * (T.dot(rho_h, self.W2) + self.by)
            R_y_total = lambda_y * self.y_data_one_hot + (1. - lambda_y) * R_y
            y_dot = R_y_total - self.y

            return [x_dot, h_dot, y_dot]

        def params_dot(Delta_x, Delta_h, Delta_y):
            rho_x = rho(self.x)
            rho_h = rho(self.h)
            rho_y = rho(self.y)

            bx_dot = T.mean(Delta_x, axis=0)
            W1_dot = (T.dot(Delta_x.T, rho_h) + T.dot(rho_x.T, Delta_h)) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            bh_dot = T.mean(Delta_h, axis=0)
            W2_dot = (T.dot(Delta_h.T, rho_y) + T.dot(rho_h.T, Delta_y)) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            by_dot = T.mean(Delta_y, axis=0)

            return [bx_dot, W1_dot, bh_dot, W2_dot, by_dot]

        lambda_x  = T.fscalar('lambda_x')
        lambda_y  = T.fscalar('lambda_y')
        epsilon_x = T.fscalar('epsilon_x')
        epsilon_h = T.fscalar('epsilon_h')
        epsilon_y = T.fscalar('epsilon_y')
        alpha_W1  = T.fscalar('alpha_W1')
        alpha_W2  = T.fscalar('alpha_W2')

        [x_dot, h_dot, y_dot] = states_dot(lambda_x, lambda_y)

        Delta_x = epsilon_x * x_dot
        Delta_h = epsilon_h * h_dot
        Delta_y = epsilon_y * y_dot

        [bx_dot, W1_dot, bh_dot, W2_dot, by_dot] = params_dot(Delta_x, Delta_h, Delta_y)

        Delta_bx = alpha_W1 * bx_dot
        Delta_W1 = alpha_W1 * W1_dot
        Delta_bh = alpha_W1 * bh_dot
        Delta_W2 = alpha_W2 * W2_dot
        Delta_by = alpha_W2 * by_dot

        x_new = self.x + Delta_x
        h_new = self.h + Delta_h
        y_new = self.y + Delta_y

        bx_new = self.bx + Delta_bx
        W1_new = self.W1 + Delta_W1
        bh_new = self.bh + Delta_bh
        W2_new = self.W2 + Delta_W2
        by_new = self.by + Delta_by

        updates = [(self.x,x_new), (self.h,h_new), (self.y,y_new), (self.bx,bx_new), (self.W1,W1_new), (self.bh,bh_new), (self.W2,W2_new), (self.by,by_new)]
        updates_relaxation = [(self.h,h_new), (self.y,y_new)]

        norm_grad_hy = T.sqrt( (h_dot ** 2).mean(axis=0).sum() + (y_dot ** 2).mean(axis=0).sum() )

        Delta_W1_relative = T.sqrt( (Delta_W1 ** 2).mean() ) / T.sqrt( (self.W1 ** 2).mean() )
        Delta_W2_relative = T.sqrt( (Delta_W2 ** 2).mean() ) / T.sqrt( (self.W2 ** 2).mean() )

        iterative_function = theano.function(
            inputs=[lambda_x, lambda_y, epsilon_x, epsilon_h, epsilon_y, alpha_W1, alpha_W2],
            outputs=[self.energy(), norm_grad_hy, self.prediction, self.error_rate, self.mse, Delta_W1_relative, Delta_W2_relative],
            updates=updates
        )

        relaxation_function = theano.function(
            inputs=[lambda_y, epsilon_h, epsilon_y],
            outputs=[self.energy(), norm_grad_hy, self.prediction, self.error_rate, self.mse],
            updates=updates_relaxation
        )

        return iterative_function, relaxation_function