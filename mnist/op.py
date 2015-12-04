import cPickle
import math
import numpy as np
import os
from sys import stdout
from theano import Op, Apply
from theano.tensor import as_tensor_variable

path = os.path.dirname(os.path.abspath(__file__))+os.sep+"op_values.save"
n_iterations = 1000
n_points = 10001
min_range = -100.
max_range = 100.
a=1.5
every = 0.1

def rho(a,x):
  return 1. / (1. + np.exp(a/2.-a*x))

def rho_prime(a,x):
    rho_x = rho(a,x)
    return a * rho_x * (1.-rho_x)

def rho_second(a,x):
    rho_x = rho(a,x)
    return a*a * rho_x * (1.-rho_x) * (1.-2.*rho_x)

def compute_values(min_range, max_range, n_points, n_iterations, plot=False):

    print("computing p_values and s_values for the operator")
    p = np.linspace(min_range, max_range, n_points)
    s = np.copy(p)
    for k in range(n_iterations):
        epsilon = 1./(1.+k)
        s = s + epsilon * (rho_prime(a,s) * p - s)
        stdout.write("\riteration=%i  " % (k))
        stdout.flush()
    stdout.write("\n")
    s_check = rho_prime(a,s) * p # s should be a fixed point, i.e. s_check = s 
    d = rho_second(a,s) * p # derivative

    indices = []
    for i in range(n_points):
        if i==0 or s[i] > s[indices[-1]] + every:
            indices.append(i)
    indices = np.array(indices)

    if plot:
        import matplotlib.pyplot as plt
        #plt.plot(p, s, "b-", linewidth = 2)
        #plt.plot(p, s_check, "g-", linewidth = 2)
        plt.plot(p, d, "g-", linewidth = 2)
        plt.plot(p[indices], s[indices], "r-", linewidth = 2)
        plt.xlim(-10.,10.)
        plt.ylim(-1.,1.)
        plt.show()

    values = [p[indices], s[indices], d[indices]]
    f = file(path, 'wb')
    cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    return values

if os.path.isfile(path):
    f = file(path, 'rb')
    [p_values, s_values, d_values] = cPickle.load(f)
    f.close()
else:
    [p_values, s_values, d_values] = compute_values(min_range, max_range, n_points, n_iterations)

class MyOp(Op):
    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = sum([(x>p1)*(x<=p2)*((p2-x)*s1+(x-p1)*s2)/(p2-p1) for p1,p2,s1,s2 in zip(p_values[:-1],p_values[1:],s_values[:-1],s_values[1:])])

    def infer_shape(self, node, i0_shapes):
        return i0_shapes

    def grad(self, inputs, output_grads):
        return [output_grads[0] * 3]