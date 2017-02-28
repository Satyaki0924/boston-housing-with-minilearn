"""
Author: Satyaki Sanyal
Base class for my mini tensor-flow
"""


class Node:
    """
    Initialising constructor
    """

    def __init__(self, inbound_nodes=[]):
        """
        The constructor is used to set properties
        for all the subclasses which inherits from this class
        """
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.gradients = {}
        self.outbound_nodes = []
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)

    def forward_propagation(self):
        """
        Every node that uses this class as a base class
        must define it's own 'forward_propagation' method
        """
        raise NotImplementedError

    def backward_propagation(self):
        """
        Every node that uses this class as a base class
        must define it's own 'backward_propagation' method
        """
        raise NotImplementedError
