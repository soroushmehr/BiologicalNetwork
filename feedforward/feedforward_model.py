import cPickle
import gzip
import numpy as np
import os
import theano
import theano.tensor as T
import theano.tensor.extra_ops

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
            W2_values = np.zeros((500,10), dtype=theano.config.floatX)
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
        self.h      = T.dot(rho(self.x_data), self.W1) + self.bh
        self.y      = T.dot(rho(self.h), self.W2) + self.by
        self.y_data = theano.shared(value=np.zeros((self.batch_size, ), dtype='int64'), name='y_data', borrow=True)

        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        self.clamp(index=0)

        self.states = [self.x_data, self.h, self.y, self.y_data_one_hot]

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

    def clamp(self, index):
        self.x_data.set_value(self.train_set_x[index*self.batch_size:(index+1)*self.batch_size,])
        self.y_data.set_value(self.train_set_y[index*self.batch_size:(index+1)*self.batch_size,])

    def build_iterative_step(self):

        [g_W2, g_by] = T.grad(cost=self.square_loss, wrt=[self.W2, self.by])
        
        lr = T.dscalar('lr')

        W2_new = self.W2 - lr * g_W2
        by_new = self.by - lr * g_by

        updates = [(self.W2,W2_new), (self.by,by_new)]

        iterative_step = theano.function(
            inputs=[lr],
            outputs=[self.prediction, self.error_rate, self.square_loss],
            updates=updates
        )

        return iterative_step