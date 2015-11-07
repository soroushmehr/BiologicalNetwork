import cPickle
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops
#from theano.tensor.shared_randomstreams import RandomStreams

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
    W_values *= 4

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
    return T.clip(x, 0., 1.) # hard-sigmoid
    # return T.nnet.sigmoid(x)

def rho_prime(x):
    return (x > 0.) * (x < 1.) # hard-sigmoid
    # y = T.nnet.sigmoid(x)
    # return y * (1 - y)

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
        train_set, valid_set, test_set = mnist()
        self.train_set_x, self.train_set_y = train_set

        # INITIALIZE STATES
        self.batch_size = batch_size
        self.rng = np.random.RandomState()

        self.x_data = theano.shared(value=np.zeros((self.batch_size, 28*28)), name='x_data', borrow=True)
        self.x      = theano.shared(value=np.zeros((self.batch_size, 28*28)), name='x',      borrow=True)
        self.h      = theano.shared(value=np.zeros((self.batch_size, 500)),   name='h',      borrow=True)
        self.y      = theano.shared(value=np.zeros((self.batch_size, 10)),    name='y',      borrow=True)
        self.y_data = theano.shared(value=np.zeros((self.batch_size, ), dtype='int64'), name='y_data', borrow=True)

        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        self.clamp(index=0)

        self.states = [self.x_data, self.x, self.h, self.y, self.y_data_one_hot]

        #self.theano_rng = RandomStreams(self.rng.randint(2 ** 30)) # will be used when introducing noise in Langevin MCMC

        self.prediction  = T.argmax(self.y, axis=1)
        self.error_rate  = T.mean(T.neq(self.prediction, self.y_data))
        self.square_loss = T.mean(((self.y - self.y_data_one_hot) ** 2).sum(axis=1))

        self.iterative_step = self.build_iterative_step()

    def save(self):
        f = file(self.path, 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def clamp(self, index, clear=True):
        self.x_data.set_value(self.train_set_x[index*self.batch_size:(index+1)*self.batch_size,])
        self.y_data.set_value(self.train_set_y[index*self.batch_size:(index+1)*self.batch_size,])

        if clear:
            self.x.set_value(self.train_set_x[index*self.batch_size:(index+1)*self.batch_size,])
            self.h.set_value(np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size, 500) ),
                dtype=theano.config.floatX
            ))
            self.y.set_value(np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size, 10) ),
                dtype=theano.config.floatX
            ))

    def energy(self):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        rho_y = rho(self.y)
        square_norm = ( T.batched_dot(self.x,self.x) + T.batched_dot(self.h,self.h) + T.batched_dot(self.y,self.y) ) / 2
        uni = - T.dot(rho_x, self.bx) - T.dot(rho_h, self.bh) - T.dot(rho_y, self.by)
        bi  = - T.batched_dot( T.dot(rho_x, self.W1), rho_h ) - T.batched_dot( T.dot(rho_h, self.W2), rho_y )
        return  square_norm+uni+bi


    def build_iterative_step(self):

        def states_dot(lambda_x, lambda_y):
            rho_x = rho(self.x)
            rho_h = rho(self.h)
            rho_y = rho(self.y)

            x_pressure = rho_prime(self.x) * (T.dot(rho_h, self.W1.T) + self.bx)
            x_pressure_final = lambda_x * self.x_data + (1 - lambda_x) * x_pressure
            x_dot = x_pressure_final - self.x

            h_pressure = rho_prime(self.h) * (T.dot(rho_x, self.W1) + T.dot(rho_y, self.W2.T) + self.bh)
            h_dot = h_pressure - self.h

            y_pressure = rho_prime(self.y) * (T.dot(rho_h, self.W2) + self.by)
            y_pressure_final = lambda_y * self.y_data_one_hot + (1 - lambda_y) * y_pressure
            y_dot = y_pressure_final - self.y

            return [x_dot, h_dot, y_dot]

        def params_delta(x_delta, h_delta, y_delta):
            rho_x = rho(self.x)
            rho_h = rho(self.h)
            rho_y = rho(self.y)

            bx_delta = T.mean(x_delta, axis=0)
            W1_delta = (T.dot(x_delta.T, rho_h) + T.dot(rho_x.T, h_delta)) / self.x.shape[0]
            bh_delta = T.mean(h_delta, axis=0)
            W2_delta = (T.dot(h_delta.T, rho_y) + T.dot(rho_h.T, y_delta)) / self.x.shape[0]
            by_delta = T.mean(y_delta, axis=0)

            return [bx_delta, W1_delta, bh_delta, W2_delta, by_delta]

        lambda_x = T.dscalar('lambda_x')
        lambda_y = T.dscalar('lambda_y')
        epsilon_x = T.dscalar('epsilon_x')
        epsilon_h = T.dscalar('epsilon_h')
        epsilon_y = T.dscalar('epsilon_y')
        epsilon_W1 = T.dscalar('epsilon_W1')
        epsilon_W2 = T.dscalar('epsilon_W2')

        [x_dot, h_dot, y_dot] = states_dot(lambda_x, lambda_y)

        x_delta = epsilon_x * x_dot
        h_delta = epsilon_h * h_dot
        y_delta = epsilon_y * y_dot

        [bx_delta, W1_delta, bh_delta, W2_delta, by_delta] = params_delta(x_delta, h_delta, y_delta)

        x_new = self.x + x_delta
        h_new = self.h + h_delta
        y_new = self.y + y_delta

        bx_new = self.bx + epsilon_W1 * bx_delta
        W1_new = self.W1 + epsilon_W1 * W1_delta
        bh_new = self.bh + epsilon_W1 * bh_delta
        W2_new = self.W2 + epsilon_W2 * W2_delta
        by_new = self.by + epsilon_W2 * by_delta

        updates = [(self.x,x_new), (self.h,h_new), (self.y,y_new), (self.bx,bx_new), (self.W1,W1_new), (self.bh,bh_new), (self.W2,W2_new), (self.by,by_new)]

        iterative_step = theano.function(
            inputs=[lambda_x, lambda_y, epsilon_x, epsilon_h, epsilon_y, epsilon_W1, epsilon_W2],
            outputs=[self.energy(), self.prediction, self.error_rate, self.square_loss],
            updates=updates
        )

        return iterative_step