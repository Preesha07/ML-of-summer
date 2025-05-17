import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))

class Neuron():
  def __init__(self,weights,bias):
    self.weights=weights
    self.bias=bias

  def feedforward(self, inputs):
    return sigmoid(np.dot(self.weights,inputs)+self.bias)

weights=np.array([0,1])
bias=4
n=Neuron(weights,bias)
x=np.array([1,3])
print(n.feedforward(x))
