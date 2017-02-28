"""
The mean squared error cost function.
Should be used as the last node for a network.
Read more about MSE: https://en.wikipedia.org/wiki/Mean_squared_error
"""
import numpy as np

from minilearn.Node import Node


class MSE(Node):
    def __init__(self, y, a):
        # Call the base class' constructor.
        Node.__init__(self, [y, a])

    def forward_propagation(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff ** 2)

    def backward_propagation(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff
