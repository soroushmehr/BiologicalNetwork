from biologicalnetwork import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# parameters for the iterativeinference
batch_size = 1000
n_inference_steps = 10
lambda_x = 1.
lambda_y = 0.
eps_s = 0.1
eps_w = 0.

net = Network(batch_size=batch_size)
inference_step = net.build_inference_step()

train_set, valid_set, test_set = mnist()
train_set_x, train_set_y = train_set

print("inference started")
net.clear(x_value=train_set_x[:batch_size,])
for k in range(n_inference_steps):
    inference_step(train_set_x[:batch_size,], train_set_y[:batch_size,], lambda_x, lambda_y, eps_s, eps_w)
print("inference finished")

# PCA
pca = PCA(n_components=2)
h = net.h.get_value()
h_2D = pca.fit(h).transform(h)

plt.figure()
plt.scatter( h_2D[:,0], h_2D[:,1] )
plt.title('PCA of configurations h (%i examples)' % (batch_size))
plt.show()