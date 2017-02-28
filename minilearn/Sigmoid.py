"""
Represents a node that performs the sigmoid activation function.
We use the formula σ = 1 / 1 + e**-x
"""
import numpy as np

from minilearn.Node import Node


class Sigmoid(Node):
    def __init__(self, node):
        """
        Initialise constructor and set node to inbound nodes
        """
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This is used in both the propagation functions
        :param x: A simple numpy array-like object
        :return: returns sigmoid of x
        """
        return 1.0 / (1.0 + np.exp(-x))

    def forward_propagation(self):
        """
        Activates the node with sigmoid.
        """
        node = self.inbound_nodes[0].value
        self.value = self._sigmoid(node)

    def backward_propagation(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialise gradients to zero.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # loop the outputs. Gradient will
        # change based on each output.
        for n in self.outbound_nodes:
            # Get partial of the cost wrt
            #  current node
            grad_cost = n.gradients[self]
            sigmoid = self.value
            # σ'(x) = σ(x) * (1 - σ(x))
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost
