import cPickle
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops
#from theano.tensor.shared_randomstreams import RandomStreams

def load(path):
    f = file(path, 'rb')
    params = cPickle.load(f)
    f.close()
    return params

def dataset():
    x = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0],[1],[1],[0]])
    return x,y

def rho(x):
    return theano.tensor.clip(x, 0., 1.)
    #return T.nnet.sigmoid(x)

def rho_prime(x):
    return (x > 0.) * (x < 1.)
    #return T.grad(rho(x),x)

class Network(object):

    def __init__(self, batch_size=1):

        self.rng = np.random.RandomState()

        self.batch_size = batch_size

        # LOAD/INITIALIZE PARAMETERS
        if not os.path.isfile("params.save"):
            bx_values = np.zeros((2,),  dtype=theano.config.floatX)
            W1_values = np.asarray(self.rng.uniform( low=-0.5, high=0.5, size=(2,3) ), dtype=theano.config.floatX)
            bh_values = np.zeros((3,),  dtype=theano.config.floatX)
            W2_values = np.asarray(self.rng.uniform( low=-0.5, high=0.5, size=(3,1) ), dtype=theano.config.floatX)
            by_values = np.zeros((1,),  dtype=theano.config.floatX)
        else:
            [bx_values, W1_values, bh_values, W2_values, by_values] = load("params.save")

        self.bx = theano.shared(value=bx_values, name='bx', borrow=True)
        self.W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        self.bh = theano.shared(value=bh_values, name='bh', borrow=True)
        self.W2 = theano.shared(value=W2_values, name='W2', borrow=True)
        self.by = theano.shared(value=by_values, name='by', borrow=True)

        self.params = [self.bx, self.W1, self.bh, self.W2, self.by]

        # LOAD DATASETS
        self.data_x, self.data_y = dataset()

        # INITIALIZE STATES
        

        self.x_data = theano.shared(value=np.zeros((batch_size,2)), name='x_data', borrow=True)
        self.x      = theano.shared(value=np.zeros((batch_size,2)), name='x',      borrow=True)
        self.h      = theano.shared(value=np.zeros((batch_size,3)), name='h',      borrow=True)
        self.y      = theano.shared(value=np.zeros((batch_size,1)), name='y',      borrow=True)
        self.y_data = theano.shared(value=np.zeros((batch_size,1)), name='y_data', borrow=True)

        self.clamp(index=0)

        self.states = [self.x_data, self.x, self.h, self.y, self.y_data]

        self.prediction  = (self.y > 0.5)
        self.error_rate  = T.mean(T.neq(self.prediction, self.y_data))
        self.square_loss = T.mean(((self.y - self.y_data) ** 2).sum(axis=1))

    def save(self):
        f = file("params.save", 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def clamp(self, index, clear=True):
        self.x_data.set_value(self.data_x[index*self.batch_size:(index+1)*self.batch_size,])
        self.y_data.set_value(self.data_y[index*self.batch_size:(index+1)*self.batch_size,])

        if clear:
            self.x.set_value(self.data_x[index*self.batch_size:(index+1)*self.batch_size,])
            self.h.set_value(np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size,3) ),
                dtype=theano.config.floatX
            ))
            self.y.set_value(np.asarray(
                self.rng.uniform( low=0, high=1, size=(self.batch_size,1) ),
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
        W1_delta = (T.dot(x_delta.T, rho_h) + T.dot(rho_x.T, h_delta)) / self.x.shape[0]
        bh_delta = T.mean(h_delta, axis=0)
        W2_delta = (T.dot(h_delta.T, rho_y) + T.dot(rho_h.T, y_delta)) / self.x.shape[0]
        by_delta = T.mean(y_delta, axis=0)

        return [bx_delta, W1_delta, bh_delta, W2_delta, by_delta]

    def build_inference_step(self):

        [x_dot, h_dot, y_dot] = self.states_dot()

        lambda_x = T.dscalar('lambda_x')
        lambda_y = T.dscalar('lambda_y')
        epsilon_states = T.dscalar('epsilon_states')
        epsilon_params = T.dscalar('epsilon_params')
        epsilon_y = T.dscalar('epsilon_y')

        x_delta = (1 - lambda_x) * epsilon_states * x_dot + lambda_x * (self.x_data - self.x)
        h_delta = epsilon_states * h_dot
        y_delta = (1 - lambda_y) * epsilon_states * y_dot + lambda_y * epsilon_y * (self.y_data - self.y)

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

        inference_step = theano.function(
            inputs=[lambda_x, lambda_y, epsilon_states, epsilon_params, epsilon_y],
            outputs=[self.energy(), self.prediction, self.error_rate, self.square_loss],
            updates=updates
        )

        return inference_step