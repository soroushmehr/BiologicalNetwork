import cPickle
import numpy as np
import os
import sys
import theano
import theano.tensor as T

path = os.path.dirname(os.path.abspath(__file__))+os.sep+os.pardir
sys.path.append(path)
from outside_world import Outside_World
from op import MyOp

my_op = MyOp()

class Network(object):

    def __init__(self, path="params.save", batch_size=1, n_hidden=500):

        self.path = path
        self.batch_size = batch_size
        states_size   = [self.n_input, self.n_hidden, self.n_output] = [28*28, n_hidden, 10]
        states_names  = ["x", "h", "y"]

        # LOAD/INITIALIZE PARAMETERS
        self.params = [self.bx, self.W1, self.bh, self.W2, self.by] = self.__load_params(path, n_hidden)
        self.params_uni = [self.bx, self.bh, self.by]
        self.params_bi = [self.W1, self.W2]

        # INITIALIZE STATES
        states_values = [np.zeros((self.batch_size, size), dtype=theano.config.floatX) for size in states_size]
        self.states   = [self.x, self.h, self.y] = [theano.shared(value=value, name=name, borrow=True) for value,name in zip(states_values,states_names)]

        # CHARACTERISTICS OF THE NETWORK
        def rho(s):
            return T.nnet.sigmoid(4.*s-2.)
        self.rho_states = [rho(state) for state in self.states]
        def energy_function():
            squared_norm =   sum( [T.batched_dot(state,state) for state in self.states] ) / 2.
            uni_terms    = - sum( [T.dot(rho_state,param_uni) for rho_state,param_uni in zip(self.rho_states,self.params_uni)] )
            bi_terms     = - sum( [T.batched_dot( T.dot(rho_1, param), rho_2) for rho_1, rho_2, param in zip(self.rho_states[:-1],self.rho_states[1:],self.params_bi)] )
            return squared_norm + uni_terms + bi_terms
        self.energy = T.mean(energy_function())
        self.energy_sum = T.sum(energy_function())
        self.prediction = T.argmax(self.y, axis=1)

        # LOAD OUTSIDE WORLD
        self.outside_world = Outside_World(batch_size)

        # THEANO FUNCTIONS
        self.initialize = self.__build_initialize_function()
        self.iterate, self.relax = self.__build_iterative_functions()

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
            [bx_values, W1_values, bh_values, W2_values, by_values] = cPickle.load(f)
            f.close()
        else:
            bx_values = np.zeros((self.n_input,),  dtype=theano.config.floatX)
            bh_values = np.zeros((self.n_hidden,), dtype=theano.config.floatX)
            by_values = np.zeros((self.n_output,), dtype=theano.config.floatX)
            W1_values = initialize_layer(self.n_input, self.n_hidden)
            W2_values = initialize_layer(self.n_hidden, self.n_output)

        params_values = [bx_values, W1_values, bh_values, W2_values, by_values]
        params_names  = ["bx", "W1", "bh", "W2", "by"]
        params = [theano.shared(value=value, name=name, borrow=True) for value, name in zip(params_values, params_names)]

        return params

    def __build_initialize_function(self):

        def rho(s):
            return T.nnet.sigmoid(4.*s-2.)

        x_init = self.outside_world.x_data                                                                           # initialize x=x_data
        # h_init = T.unbroadcast(T.constant(np.zeros((self.batch_size, self.n_hidden), dtype=theano.config.floatX)),0) # initialize h=0
        # y_init = T.unbroadcast(T.constant(np.zeros((self.batch_size, self.n_output), dtype=theano.config.floatX)),0) # initialize y=0
        h_init = my_op(2 * (T.dot(rho(x_init), self.W1) + self.bh))                                                   # initialize h by forward propagation
        y_init = my_op(T.dot(rho(h_init), self.W2) + self.by)                                                         # initialize y by forward propagation
        states_init = [x_init, h_init, y_init]

        initialize = theano.function(
            inputs=[],
            outputs=[],
            updates=zip(self.states,states_init)
        )

        return initialize

    def __build_iterative_functions(self):

        def states_dot(lambda_x, lambda_y, x_data, y_data):
            [x_dot, h_dot, y_dot] = T.grad(-self.energy_sum, self.states)
            x_dot_final = lambda_x * (x_data - self.x) + (1. - lambda_x) * x_dot
            y_dot_final = lambda_y * (y_data - self.y) + (1. - lambda_y) * y_dot
            return [x_dot_final, h_dot, y_dot_final]

        lambda_x = T.fscalar('lambda_x')
        lambda_y = T.fscalar('lambda_y')

        x_data = self.outside_world.x_data
        y_data = self.outside_world.y_data_one_hot

        states_dot = [x_dot, h_dot, y_dot] = states_dot(lambda_x, lambda_y, x_data, y_data)

        kinetic_energy = T.mean( sum( [(state_dot ** 2).sum(axis=1) for state_dot in states_dot] ) )
        params_dot = T.grad(kinetic_energy, self.params)

        # UPDATES
        epsilon  = T.fscalar('epsilon')
        alpha_W1 = T.fscalar('alpha_W1')
        alpha_W2 = T.fscalar('alpha_W2')
        learning_rates = [alpha_W1,alpha_W1,alpha_W1,alpha_W2,alpha_W2]

        Delta_states = [epsilon * state_dot for state_dot in states_dot]
        Delta_params = [alpha * param_dot for alpha,param_dot in zip(learning_rates,params_dot)]
        states_new = [state+Delta for state,Delta in zip(self.states,Delta_states)]
        params_new = [param+Delta for param,Delta in zip(self.params,Delta_params)]
        updates_states = zip(self.states,states_new)
        updates_params = zip(self.params,params_new)

        # OUTPUTS FOR MONITORING
        error_rate   = T.mean(T.neq(self.prediction, self.outside_world.y_data))
        mse          = T.mean(((self.y - self.outside_world.y_data_one_hot) ** 2).sum(axis=1))
        norm_grad_hy = T.sqrt( (h_dot ** 2).mean(axis=0).sum() + (y_dot ** 2).mean(axis=0).sum() )
        Delta_W1 = Delta_params[1]
        Delta_W2 = Delta_params[3]
        Delta_logW1  = T.sqrt( (Delta_W1 ** 2).mean() ) / T.sqrt( (self.W1 ** 2).mean() )
        Delta_logW2  = T.sqrt( (Delta_W2 ** 2).mean() ) / T.sqrt( (self.W2 ** 2).mean() )

        # THEANO FUNCTIONS
        iterative_function = theano.function(
            inputs=[lambda_x, lambda_y, epsilon, alpha_W1, alpha_W2],
            outputs=[self.energy, norm_grad_hy, self.prediction, error_rate, mse, Delta_logW1, Delta_logW2],
            updates=updates_params+updates_states
        )

        relaxation_function = theano.function(
            inputs=[epsilon],
            outputs=[self.energy, norm_grad_hy, self.prediction, error_rate, mse],
            givens={
            lambda_y: T.constant(0.)
            },
            updates=updates_states[1:3]
        )

        return iterative_function, relaxation_function