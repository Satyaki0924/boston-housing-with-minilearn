"""
Test Mini-learn on Boston housing data-set.
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import resample

from minilearn import *

# Load data
data = load_boston()
X_ = data['data']
y_ = data['target']

# Normalize data
X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

n_features = X_.shape[1]
# TODO: set hidden layers
n_hidden = 100
W1_ = np.random.randn(n_features, n_hidden)
b1_ = np.zeros(n_hidden)
W2_ = np.random.randn(n_hidden, 1)
b2_ = np.zeros(1)

# Neural network
X, y = Input.Input(), Input.Input()
W1, b1 = Input.Input(), Input.Input()
W2, b2 = Input.Input(), Input.Input()

l1 = Linear.Linear(X, W1, b1)
s1 = Sigmoid.Sigmoid(l1)
l2 = Linear.Linear(s1, W2, b2)
cost = MSE.MSE(y, l2)

feed_dict = {
    X: X_,
    y: y_,
    W1: W1_,
    b1: b1_,
    W2: W2_,
    b2: b2_
}
"""
set epoch and batch size here:
"""
# TODO: Set epoch
epochs = 1000
# Total number of examples
m = X_.shape[0]
# TODO: Set Batch size
batch_size = 6
steps_per_epoch = m // batch_size

graph = Functions.Functions.topological_sort(feed_dict)
trainable = [W1, b1, W2, b2]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

        # Reset value of X and y Inputs
        X.value = X_batch
        y.value = y_batch

        # Step 2
        Functions.Functions.forward_and_backward_pass(graph)

        # Step 3
        Functions.Functions.sgd_update(trainable)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
