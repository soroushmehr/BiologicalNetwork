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
    return theano.tensor.clip(x, 0., 1.)

def rho_prime(x):
    return (x > 0.) * (x < 1.)

class Network(object):

    def __init__(self, batch_size=1):

        if not os.path.isfile("params.save"):
            bx_values = np.zeros((28*28,), dtype=theano.config.floatX)
            W1_values = initialize_layer(28*28, 500)
            bh_values = np.zeros((500,), dtype=theano.config.floatX)
            W2_values = initialize_layer(500, 10)
            by_values = np.zeros((10,), dtype=theano.config.floatX)
        else:
            [bx_values, W1_values, bh_values, W2_values, by_values] = load("params.save")

        self.batch_size = batch_size
        self.rng = np.random.RandomState()
        x_values = np.asarray(
            self.rng.uniform( low=0, high=1, size=(self.batch_size, 28*28) ),
            dtype=theano.config.floatX
        )
        h_values = np.asarray(
            self.rng.uniform( low=0, high=1, size=(self.batch_size, 500) ),
            dtype=theano.config.floatX
        )
        y_values = np.asarray(
            self.rng.uniform( low=0, high=0.2, size=(self.batch_size, 10) ),
            dtype=theano.config.floatX
        )

        self.bx = theano.shared(value=bx_values, name='bx', borrow=True)
        self.W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        self.bh = theano.shared(value=bh_values, name='bh', borrow=True)
        self.W2 = theano.shared(value=W2_values, name='W2', borrow=True)
        self.by = theano.shared(value=by_values, name='by', borrow=True)

        self.x = theano.shared(value=x_values, name='x', borrow=True)
        self.h = theano.shared(value=h_values, name='h', borrow=True)
        self.y = theano.shared(value=y_values, name='y', borrow=True)

        self.params = [self.bx, self.W1, self.bh, self.W2, self.by]
        self.states = [self.x, self.h, self.y]

        #self.theano_rng = RandomStreams(np.random.RandomState().randint(2 ** 30))

    def save(self):
        f = file("params.save", 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def clear(self, x_value = None):
        if x_value == None:
            x_value = np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size, 28*28) ),
                dtype=theano.config.floatX
            )
        self.x.set_value(x_value)
        self.h.set_value(np.asarray(
            self.rng.uniform( low=0, high=1, size=(self.batch_size, 500) ),
            dtype=theano.config.floatX
        ))
        self.y.set_value(np.asarray(
            self.rng.uniform( low=0, high=1, size=(self.batch_size, 10) ),
            dtype=theano.config.floatX
        ))

    def prediction(self):
        return T.argmax(self.y, axis=1)

    def energy(self):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        rho_y = rho(self.y)
        square_norm = ( T.batched_dot(self.x,self.x) + T.batched_dot(self.h,self.h) + T.batched_dot(self.y,self.y) ) / 2
        uni = - T.dot(rho_x, self.bx) - T.dot(rho_h, self.bh) - T.dot(rho_y, self.by)
        bi  = - T.batched_dot( T.dot(rho_x, self.W1), rho_h ) - T.batched_dot( T.dot(rho_h, self.W2), rho_y )
        return  square_norm+uni+bi

    def states_dot(self):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        rho_y = rho(self.y)

        x_pressure = rho_prime(self.x) * (T.dot(rho_h, self.W1.T) + self.bx)
        x_dot = x_pressure - self.x
        h_pressure = rho_prime(self.h) * (T.dot(rho_x, self.W1) + T.dot(rho_y, self.W2.T) + self.bh)
        h_dot = h_pressure - self.h
        y_pressure = rho_prime(self.y) * (T.dot(rho_h, self.W2) + self.by)
        y_dot = y_pressure - self.y

        return [x_dot, h_dot, y_dot]

    def params_delta(self, x_delta, h_delta, y_delta):
        rho_x = rho(self.x)
        rho_h = rho(self.h)
        rho_y = rho(self.y)

        bx_delta = T.mean(x_delta, axis=0)
        W1_delta = T.dot(x_delta.T, rho_h) + T.dot(rho_x.T, h_delta) / self.x.shape[0]
        bh_delta = T.mean(h_delta, axis=0)
        W2_delta = T.dot(h_delta.T, rho_y) + T.dot(rho_h.T, y_delta) / self.x.shape[0]
        by_delta = T.mean(y_delta, axis=0)

        return [bx_delta, W1_delta, bh_delta, W2_delta, by_delta]

    def build_inference_step(self):

        [x_dot, h_dot, y_dot] = self.states_dot()

        x_data = T.matrix('x')
        y_data = T.lvector('y')
        y_data_one_hot = T.extra_ops.to_one_hot(y_data, 10)

        lambda_x = T.dscalar('lambda_x')
        lambda_y = T.dscalar('lambda_y')
        epsilon_states = T.dscalar('epsilon_states')
        epsilon_params = T.dscalar('epsilon_params')

        x_delta = (1 - lambda_x) * epsilon_states * x_dot + lambda_x * (x_data - self.x)
        h_delta = epsilon_states * h_dot
        y_delta = (1 - lambda_y) * epsilon_states * y_dot + lambda_y * (y_data_one_hot - self.y)

        [bx_delta, W1_delta, bh_delta, W2_delta, by_delta] = self.params_delta(x_delta, h_delta, y_delta)

        x_new = self.x + x_delta
        h_new = self.h + h_delta
        y_new = self.y + y_delta

        bx_new = self.bx + epsilon_params * bx_delta
        W1_new = self.W1 + epsilon_params * W1_delta
        bh_new = self.bh + epsilon_params * bh_delta
        W2_new = self.W2 + epsilon_params * W2_delta
        by_new = self.by + epsilon_params * by_delta

        updates = [(self.x,x_new), (self.h,h_new), (self.y,y_new), (self.bx,bx_new), (self.W1,W1_new), (self.bh,bh_new), (self.W2,W2_new), (self.by,by_new)]
        energy = self.energy()
        prediction = self.prediction()
        error = T.mean(T.neq(y_data, prediction))

        inference_step = theano.function(
            inputs=[x_data, y_data, lambda_x, lambda_y, epsilon_states, epsilon_params],
            outputs=[energy, prediction, error],
            updates=updates
        )

        return inference_step