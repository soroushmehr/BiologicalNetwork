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

def rho(x):
  return 1. / (1. + math.exp(2.-4.*x))

def rho_prime(x):
    rho_x = rho(x)
    return 4. * rho_x * (1.-rho_x)

def compute_fixed_point(p): # p = pression, s = fixed point
    s = p
    for k in range(n_iterations):
        epsilon = 1./(1.+k)
        s = s + epsilon * (rho_prime(s) * p - s)
    s_check = rho_prime(s) * p # s should be afixed point, i.e. s_check = s 
    return s, s_check

def compute_values(plot=False):
    p = np.linspace(min_range, max_range, n_points)
    s = np.zeros(n_points)
    s_check = np.zeros(n_points)

    p_values = []
    s_values = []

    print("computing p_values and s_values for the operator")

    for i in range(n_points):
        p_value = p[i]
        s[i], s_check[i] = compute_fixed_point(p_value)
        if len(s_values)==0 or s[i] > s_values[-1] + 0.1:
            p_values.append(p_value)
            s_values.append(s[i])
        stdout.write("\rp_value=%.2f  " % (p_value))
        stdout.flush()
    stdout.write("\n")

    if plot:
        import matplotlib.pyplot as plt
        #plt.plot(p, s, "b-", linewidth = 2)
        #plt.plot(p, s_check, "g-", linewidth = 2)
        plt.plot(np.array(p_values), np.array(s_values), "r-", linewidth = 2)
        plt.show()

    values = [p_values, s_values]
    f = file(path, 'wb')
    cPickle.dump(values, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

    return values

if os.path.isfile(path):
    f = file(path, 'rb')
    [p_values, s_values] = cPickle.load(f)
    f.close()
else:
    [p_values, s_values] = compute_values()

'''q_values = [np.float32(-100.),np.float32(-60.44),np.float32(-36.12),np.float32(-21.28),np.float32(-12.3),
np.float32(-6.94),np.float32(-3.78),np.float32(-1.94),np.float32(-0.88),np.float32(-0.23),
np.float32(0.02),np.float32(0.2),np.float32(0.3),np.float32(0.38),np.float32(0.46),
np.float32(0.58),np.float32(0.76),np.float32(1.04),np.float32(1.48),np.float32(2.18),
np.float32(3.3),np.float32(5.08),np.float32(7.9),np.float32(12.36),np.float32(19.42),
np.float32(30.54),np.float32(48.04),np.float32(75.5)]

h_values = [np.float32(-0.99729261864835617),np.float32(-0.89723086628259308),np.float32(-0.79717397878914287),np.float32(-0.69708246612205638),np.float32(-0.59682676550115632),
np.float32(-0.49666378742140538),np.float32(-0.396568808405038),np.float32(-0.29617133639974386),np.float32(-0.19417364632638415),np.float32(-0.093500562854766267),
np.float32(0.0086223049042782121),np.float32(0.117083800518579),np.float32(0.22551233426881651),np.float32(0.34643803638462772),np.float32(0.45654416817163457),
np.float32(0.56907070512214353),np.float32(0.67447901907693208),np.float32(0.77674040447169557),np.float32(0.87745174543119275),np.float32(0.97804951372252047),
np.float32(1.0789966826038837),np.float32(1.179655288666364),np.float32(1.2799723149504749),np.float32(1.3800749013953804),np.float32(1.4802792188173179),
np.float32(1.5803273873428634),np.float32(1.6803941222793461),np.float32(1.7804202646655001)]'''

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