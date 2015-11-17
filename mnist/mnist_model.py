import cPickle
import gzip
import numpy as np
import os
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import theano.tensor.extra_ops
from theano.tensor.shared_randomstreams import RandomStreams

class Outside_World(object):

    def __init__(self, batch_size):

        self.batch_size = batch_size
        self.train_set_size = 50000
        self.valid_set_size = 10000
        self.test_set_size = 10000

        # LOAD MNIST DATASET
        f = gzip.open("mnist.pkl.gz", 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        train_set_x, train_set_y = shared_dataset(train_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        test_set_x,  test_set_y  = shared_dataset(test_set)

        # STATE OF THE OUTSIDE WORLD
        #np.ones(1, dtype=np.int64)
        self.dataset = theano.shared(np.int64(1), name='dataset') # dataset=1 for train_set; dataset=2 for valid_set; dataset=3 for test_set
        self.index  = theano.shared(np.int64(0), name='index')

        temp_x = ifelse(T.eq(self.dataset,1), train_set_x, test_set_x)
        temp_y = ifelse(T.eq(self.dataset,1), train_set_y, test_set_y)

        set_x = ifelse(T.eq(self.dataset,2), valid_set_x, temp_x)
        set_y = ifelse(T.eq(self.dataset,2), valid_set_y, temp_y)

        self.x_data = set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        self.y_data = set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        # THEANO FUNCTION TO UPDATE THE STATE
        self.set = self.build_set_function()

    def build_set_function(self):

        index_new = T.lscalar('index_new')
        dataset_new = T.lscalar('dataset_new')

        updates = [(self.index, index_new), (self.dataset, dataset_new)]

        set_function = theano.function(
            inputs=[index_new, dataset_new],
            outputs=[],
            updates=updates
        )

        return set_function

class Network(object):

    def __init__(self, path="params.save", batch_size=1, n_hidden=500):

        self.path = path
        self.batch_size = batch_size
        self.n_input  = 28*28
        self.n_hidden = n_hidden
        self.n_output = 10

        # LOAD/INITIALIZE PARAMETERS
        self.params = [self.bx, self.W1, self.bh, self.W2, self.by] = self.load_params(path, n_hidden)

        # INITIALIZE STATES
        self.x = theano.shared(value=np.zeros((self.batch_size, self.n_input),  dtype=theano.config.floatX), name='x', borrow=True)
        self.h = theano.shared(value=np.zeros((self.batch_size, self.n_hidden), dtype=theano.config.floatX), name='h', borrow=True)
        self.y = theano.shared(value=np.zeros((self.batch_size, self.n_output), dtype=theano.config.floatX), name='y', borrow=True)

        def rho(s):
            # return T.clip(s, 0., 1.)       # hard-sigmoid
            return T.nnet.sigmoid(4.*s-2.) # sigmoid
            # return T.tanh(s)               # hyperbolic tangent

        def rho_prime(s):
            # return (s > 0.) * (s < 1.)  # hard-sigmoid
            r = T.nnet.sigmoid(4.*s-2.) # sigmoid
            return 4. * r * (1. - r)    # sigmoid
            # r = T.tanh(s)               # hyperbolic tangent
            # return 1. - r ** 2          # hyperbolic tangent

        self.rho_x = rho(self.x)
        self.rho_h = rho(self.h)
        self.rho_y = rho(self.y)

        self.rho_prime_x = rho_prime(self.x)
        self.rho_prime_h = rho_prime(self.h)
        self.rho_prime_y = rho_prime(self.y)

        # LOAD OUTSIDE WORLD
        self.outside_world = Outside_World(batch_size)

        # CHARACTERISTICS OF THE NETWORK
        self.prediction = T.argmax(self.y, axis=1)

        def energy_function():
            squared_norm = ( T.batched_dot(self.x,self.x) + T.batched_dot(self.h,self.h) + T.batched_dot(self.y,self.y) ) / 2.
            uni_terms    = - T.dot(self.rho_x, self.bx) - T.dot(self.rho_h, self.bh) - T.dot(self.rho_y, self.by)
            bi_terms     = - T.batched_dot( T.dot(self.rho_x, self.W1), self.rho_h ) - T.batched_dot( T.dot(self.rho_h, self.W2), self.rho_y )
            return squared_norm + uni_terms + bi_terms

        self.energy = energy_function()

        # THEANO FUNCTIONS
        rng = np.random.RandomState()
        self.theano_rng = RandomStreams(rng.randint(2 ** 30)) # used to initialize h and y at random at the beginning of the x-clamped relaxation phase. will also be used when introducing noise in Langevin MCMC

        self.initialize = self.build_initialize_function()
        self.iterate, self.relax = self.build_iterative_functions()

    def load_params(self, path, n_hidden):

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

        if os.path.isfile(self.path):
            f = file(path, 'rb')
            [bx_values, W1_values, bh_values, W2_values, by_values] = cPickle.load(f)
            f.close()
        else:
            bx_values = np.zeros((self.n_input,),  dtype=theano.config.floatX)
            W1_values = initialize_layer(self.n_input, self.n_hidden)
            bh_values = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
            W2_values = initialize_layer(self.n_hidden, self.n_output)
            by_values = np.zeros((self.n_output,), dtype=theano.config.floatX)

        bx = theano.shared(value=bx_values, name='bx', borrow=True)
        W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        bh = theano.shared(value=bh_values, name='bh', borrow=True)
        W2 = theano.shared(value=W2_values, name='W2', borrow=True)
        by = theano.shared(value=by_values, name='by', borrow=True)

        return [bx, W1, bh, W2, by]

    def save_params(self):
        f = file(self.path, 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def build_initialize_function(self):

        x_init = self.outside_world.x_data                                                                # initialize by clamping x_data
        h_init = T.unbroadcast(T.constant(np.zeros((self.batch_size, self.n_hidden), dtype=theano.config.floatX)),0)       # initialize h=0 and y=0
        y_init = T.unbroadcast(T.constant(np.zeros((self.batch_size, self.n_output), dtype=theano.config.floatX)),0)       # initialize h=0 and y=0
        # h_init = self.theano_rng.uniform(size=self.h.shape, low=0., high=.01, dtype=theano.config.floatX) # initialize h and y at random
        # y_init = self.theano_rng.uniform(size=self.y.shape, low=0., high=.01, dtype=theano.config.floatX) # initialize h and y at random
        # h_init = T.dot(rho(x_init), self.W1) + self.bh                                                    # initialize h and y by forward propagation
        # y_init = T.dot(rho(h_init), self.W2) + self.by                                                    # initialize h and y by forward propagation

        updates_states = [(self.x, x_init), (self.h, h_init), (self.y, y_init)]

        initialize = theano.function(
            inputs=[],
            outputs=[],
            updates=updates_states
        )

        return initialize

    def build_iterative_functions(self):

        def states_dot(lambda_x, lambda_y, x_data, y_data):
            R_x = self.rho_prime_x * (T.dot(self.rho_h, self.W1.T) + self.bx)
            R_x_total = lambda_x * x_data + (1. - lambda_x) * R_x
            x_dot = R_x_total - self.x

            R_h = self.rho_prime_h * (T.dot(self.rho_x, self.W1) + T.dot(self.rho_y, self.W2.T) + self.bh)
            h_dot = R_h - self.h

            R_y = self.rho_prime_y * (T.dot(self.rho_h, self.W2) + self.by)
            R_y_total = lambda_y * y_data + (1. - lambda_y) * R_y
            y_dot = R_y_total - self.y

            return [x_dot, h_dot, y_dot]

        def params_dot(Delta_x, Delta_h, Delta_y):
            bx_dot = T.mean(Delta_x, axis=0)
            W1_dot_fwd = T.dot(Delta_x.T, self.rho_h) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            W1_dot_bwd = T.dot(self.rho_x.T, Delta_h) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            bh_dot = T.mean(Delta_h, axis=0)
            W2_dot_fwd = T.dot(Delta_h.T, self.rho_y) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            W2_dot_bwd = T.dot(self.rho_h.T, Delta_y) / T.cast(self.x.shape[0], dtype=theano.config.floatX)
            by_dot = T.mean(Delta_y, axis=0)

            return [bx_dot, W1_dot_fwd, W1_dot_bwd, bh_dot, W2_dot_fwd, W2_dot_bwd, by_dot]

        lambda_x  = T.fscalar('lambda_x')
        lambda_y  = T.fscalar('lambda_y')
        epsilon_x = T.fscalar('epsilon_x')
        epsilon_h = T.fscalar('epsilon_h')
        epsilon_y = T.fscalar('epsilon_y')
        alpha_W1  = T.fscalar('alpha_W1')
        alpha_W2  = T.fscalar('alpha_W2')

        x_data = self.outside_world.x_data
        y_data = self.outside_world.y_data_one_hot

        [x_dot, h_dot, y_dot] = states_dot(lambda_x, lambda_y, x_data, y_data)

        Delta_x = epsilon_x * x_dot
        Delta_h = epsilon_h * h_dot
        Delta_y = epsilon_y * y_dot

        [bx_dot, W1_dot_fwd, W1_dot_bwd, bh_dot, W2_dot_fwd, W2_dot_bwd, by_dot] = params_dot(Delta_x, Delta_h, Delta_y)

        Delta_bx     = alpha_W1 * bx_dot
        Delta_W1_fwd = alpha_W1 * W1_dot_fwd
        Delta_W1_bwd = alpha_W1 * W1_dot_bwd
        Delta_W1     = Delta_W1_fwd + Delta_W1_bwd
        Delta_bh     = alpha_W1 * bh_dot
        Delta_W2_fwd = alpha_W2 * W2_dot_fwd
        Delta_W2_bwd = alpha_W2 * W2_dot_bwd
        Delta_W2     = Delta_W2_fwd + Delta_W2_bwd
        Delta_by     = alpha_W2 * by_dot

        x_new = self.x + Delta_x
        h_new = self.h + Delta_h
        y_new = self.y + Delta_y

        bx_new = self.bx + Delta_bx
        W1_new = self.W1 + Delta_W1
        bh_new = self.bh + Delta_bh
        W2_new = self.W2 + Delta_W2
        by_new = self.by + Delta_by

        # OUTPUTS FOR MONITORING
        energy_mean  = T.mean(self.energy)
        error_rate   = T.mean(T.neq(self.prediction, self.outside_world.y_data))
        mse          = T.mean(((self.y - self.outside_world.y_data_one_hot) ** 2).sum(axis=1))
        norm_grad_hy = T.sqrt( (h_dot ** 2).mean(axis=0).sum() + (y_dot ** 2).mean(axis=0).sum() )
        Delta_logW1_fwd = T.sqrt( (Delta_W1_fwd ** 2).mean() ) / T.sqrt( (self.W1 ** 2).mean() )
        Delta_logW1_bwd = T.sqrt( (Delta_W1_bwd ** 2).mean() ) / T.sqrt( (self.W1 ** 2).mean() )
        Delta_logW2_fwd = T.sqrt( (Delta_W2_fwd ** 2).mean() ) / T.sqrt( (self.W2 ** 2).mean() )
        Delta_logW2_bwd = T.sqrt( (Delta_W2_bwd ** 2).mean() ) / T.sqrt( (self.W2 ** 2).mean() )

        # UPDATES
        updates_params = [(self.bx,bx_new), (self.W1,W1_new), (self.bh,bh_new), (self.W2,W2_new), (self.by,by_new)]
        updates_states = [(self.x,x_new), (self.h,h_new), (self.y,y_new)]

        # THEANO FUNCTIONS
        iterative_function = theano.function(
            inputs=[lambda_x, lambda_y, epsilon_x, epsilon_h, epsilon_y, alpha_W1, alpha_W2],
            outputs=[energy_mean, norm_grad_hy, self.prediction, error_rate, mse, Delta_logW1_fwd, Delta_logW2_fwd, Delta_logW1_bwd, Delta_logW2_bwd],
            updates=updates_params+updates_states
        )

        relaxation_function = theano.function(
            inputs=[epsilon_h, epsilon_y],
            outputs=[energy_mean, norm_grad_hy, self.prediction, error_rate, mse],
            givens={
            lambda_y: T.constant(0.)
            },
            updates=updates_states[1:3]
        )

        return iterative_function, relaxation_function