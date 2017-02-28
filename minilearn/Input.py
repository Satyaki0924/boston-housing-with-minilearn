"""
The Input class with which we will input the data to mini-learn
"""
from minilearn.Node import Node


class Input(Node):
    def __init__(self):
        # This class constructor has to run to set all
        # the properties here.
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Node.__init__(self)

    def forward_propagation(self):
        # Nothing is to be calculated here
        pass

    def backward_propagation(self):
        # This is the input class. So the gradient
        # must be zero.
        # self is a reference to this object
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so we need to sum
        # the gradient from output gradients.
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]
