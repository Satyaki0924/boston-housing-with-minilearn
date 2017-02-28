from minilearn.Input import Input
from minilearn.Node import Node


class Functions(Node):
    """
    A class consisting of several static functions
    """

    def __init__(self):
        Node.__init__(self)

    @staticmethod
    def topological_sort(feed_dict):
        """
        Sort the nodes in topological order using Kahn's Algorithm.
        :param feed_dict: A dictionary where the key is a `Input`
        Node and the value is the respective value feed to that Node.
        :return: List of sorted nodes.
        """
        input_nodes = [n for n in feed_dict.keys()]
        G = {}
        nodes = [n for n in input_nodes]
        while len(nodes) > 0:
            n = nodes.pop(0)
            if n not in G:
                G[n] = {'in': set(), 'out': set()}
            for m in n.outbound_nodes:
                if m not in G:
                    G[m] = {'in': set(), 'out': set()}
                G[n]['out'].add(m)
                G[m]['in'].add(n)
                nodes.append(m)
        L = []
        S = set(input_nodes)
        while len(S) > 0:
            n = S.pop()

            if isinstance(n, Input):
                n.value = feed_dict[n]

            L.append(n)
            for m in n.outbound_nodes:
                G[n]['out'].remove(m)
                G[m]['in'].remove(n)
                # if no other incoming edges add to S
                if len(G[m]['in']) == 0:
                    S.add(m)
        return L

    @staticmethod
    def forward_and_backward_pass(graph):
        """
        Performs a forward pass and a backward pass through a list of sorted Nodes.
        :param graph: The result of calling `topological_sort`.
        """
        # Forward pass
        for n in graph:
            n.forward_propagation()

        # Backward pass
        for n in graph[::-1]:
            n.backward_propagation()

    @staticmethod
    def sgd_update(trainable, learning_rate=1e-2):
        """
        Updates the value of each trainable with Stochastic gradient descent
        :param trainable: A list of `Input` Nodes representing weights/biases.
        :param learning_rate: The learning rate.
        """
        # Performs SGD
        # Loop over trainable
        for t in trainable:
            # Change the trainable's value by subtracting the learning rate
            # multiplied by the partial of the cost with respect to this
            # trainable.
            partial = t.gradients[t]
            t.value -= learning_rate * partial
