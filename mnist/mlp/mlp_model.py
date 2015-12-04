import cPickle
import numpy as np
import os
import sys
import theano
import theano.tensor as T

path = os.path.dirname(os.path.abspath(__file__))+os.sep+os.pardir
sys.path.append(path)
from outside_world import Outside_World

a=3.

def rho(s):
    # return T.clip(s, 0., 1.)       # hard-sigmoid
    return T.nnet.sigmoid(a*s-a/2.)

class Network(object):

    def __init__(self, path="params.save", batch_size=1, n_hidden=500):

        self.path = path
        self.batch_size = batch_size
        self.n_input  = 28*28
        self.n_hidden = n_hidden
        self.n_output = 10

        # LOAD/INITIALIZE PARAMETERS
        self.params = [self.W1, self.bh, self.W2, self.by] = self.__load_params(path, n_hidden)

        # LOAD OUTSIDE WORLD
        self.outside_world = Outside_World(batch_size)

        # THEANO FUNCTIONS
        [self.predict, self.train] = self.__build_functions()

    def save_params(self):
        f = file(self.path, 'wb')
        params = [param.get_value() for param in self.params]
        cPickle.dump(params, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def __load_params(self, path, n_hidden):

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
            [W1_values, bh_values, W2_values, by_values] = cPickle.load(f)
            f.close()
        else:
            W1_values = initialize_layer(self.n_input, self.n_hidden)
            bh_values = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
            W2_values = initialize_layer(self.n_hidden, self.n_output)
            by_values = np.zeros((self.n_output,), dtype=theano.config.floatX)

        W1 = theano.shared(value=W1_values, name='W1', borrow=True)
        bh = theano.shared(value=bh_values, name='bh', borrow=True)
        W2 = theano.shared(value=W2_values, name='W2', borrow=True)
        by = theano.shared(value=by_values, name='by', borrow=True)

        return [W1, bh, W2, by]

    def __build_functions(self):

        x = self.outside_world.x_data
        h = T.dot(rho(x), self.W1) + self.bh
        y = T.dot(rho(h), self.W2) + self.by

        prediction = T.argmax(y, axis=1)

        error_rate = T.mean(T.neq(prediction, self.outside_world.y_data))
        mse        = T.mean(((y - self.outside_world.y_data_one_hot) ** 2).sum(axis=1))

        predict_function = theano.function(
            inputs=[],
            outputs=[error_rate, mse]
        )

        grads = T.grad(mse,self.params)

        alpha = T.fscalar('alpha')
        updates_params = [(param, param - alpha * grad) for param,grad in zip(self.params,grads)]

        train_function = theano.function(
            inputs=[alpha],
            outputs=[error_rate, mse],
            updates=updates_params
        )

        return [predict_function, train_function]