import numpy as np
def sigmoid(x):
  return 1/(1+np.exp(-x))

class Neuron():
  def __init__(self,weights,bias):
    self.weights=weights
    self.bias=bias

  def feedforward(self, inputs):
    return sigmoid(np.dot(self.weights,inputs)+self.bias)
    
x=np.array([2,3])

class Neural_Network():
  def __init__(self, weights=[0,1],bias=0):
    self.weights=weights
    self.bias=bias

    self.h2=Neuron(self.weights,self.bias)
    self.h1=Neuron(self.weights,self.bias)
    self.o1=Neuron(self.weights,self.bias)

  def ans(self,x):
    self.ans=self.o1.feedforward(np.array([self.h1.feedforward(x),self.h2.feedforward(x)]))
    return self.ans

nn= Neural_Network()
print(nn.ans(x))
