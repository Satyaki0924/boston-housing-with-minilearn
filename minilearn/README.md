# Miniflow
## A Miniature tensorflow made for experimentation
## Classes:

### Node: 
Parent class. The constructor is used to set properties for all the subclasses which inherits from this class.

### Input:
The Input class with which we will input the data to mini-learn

### Linear:
performs linear transformation using the formula Y = mX + b

### Sigmoid:
Represents a node that performs the sigmoid activation function. We use the formula σ = 1 / 1 + e**-x

### Mean Squared Error (MSE):
The mean squared error cost function. More about MSE: https://en.wikipedia.org/wiki/Mean_squared_error

### Functions:
A class consisting of several static functions:
#### topological_sort:
Sort the nodes in topological order using Kahn's Algorithm.
#### forward_and_backward_pass:
Performs a forward pass and a backward pass through a list of sorted Nodes.
#### sgd_update (Stochastic gradient descent):
Updates the value of each trainable with Stochastic gradient descent

#### Author: Satyaki Sanyal
This project must strictly be used for educational purposes only
