import cPickle
import gzip
import numpy as np
import os
import sys
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import theano.tensor.extra_ops

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path)

class Outside_World(object):

    def __init__(self, batch_size):

        self.batch_size = batch_size

        # LOAD MNIST DATASET
        if not os.path.isfile(self.path):
            import urllib
            origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, self.path)
        f = gzip.open(self.path, 'rb')
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
        temp_x = ifelse(T.eq(self.__dataset,1), train_set_x, test_set_x)
        temp_y = ifelse(T.eq(self.__dataset,1), train_set_y, test_set_y)

        final_x = ifelse(T.eq(self.__dataset,2), valid_set_x, temp_x)
        final_y = ifelse(T.eq(self.__dataset,2), valid_set_y, temp_y)

        self.x_data = final_x[self.__index * self.batch_size: (self.__index + 1) * self.batch_size]
        self.y_data = final_y[self.__index * self.batch_size: (self.__index + 1) * self.batch_size]
        self.y_data_one_hot = T.extra_ops.to_one_hot(self.y_data, 10)

        # THEANO FUNCTION TO UPDATE THE STATE
        self.set = self.build_set_function()

    path = dir_path+"\mnist.pkl.gz"
    train_set_size = 50000
    valid_set_size = 10000
    test_set_size = 10000

    __dataset = theano.shared(np.int64(1), name='dataset') # __dataset=1 for train_set; __dataset=2 for valid_set; __dataset=3 for test_set
    __index  = theano.shared(np.int64(0), name='index')

    def build_set_function(self):

        index_new = T.lscalar('index_new')
        dataset_new = T.lscalar('dataset_new')

        updates = [(self.__index, index_new), (self.__dataset, dataset_new)]

        set_function = theano.function(
            inputs=[index_new, dataset_new],
            outputs=[],
            updates=updates
        )

        return set_function