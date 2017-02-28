"""
performs linear transformation using the formula
# Y = mX + b
"""
import numpy as np

from minilearn.Node import Node


class Linear(Node):
    def __init__(self, X, W, b):
        # The Node constructor. Weight and bias
        # are treated like inbound node
        Node.__init__(self, [X, W, b])

    def forward_propagation(self):
        """
        performs operation based on Y = mX + b
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.value = np.dot(X, W) + b

    def backward_propagation(self):
        """
        calculates gradient based on output
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # loop the outputs. Gradient will
        # change based on each output.
        for n in self.outbound_nodes:
            # Get partial of the cost wrt current node
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)
