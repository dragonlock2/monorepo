import numpy as np
from abc import ABC, abstractmethod

rng = np.random.default_rng(seed=69)

class Layer(ABC): # super unoptimized API
    @abstractmethod
    def forward(self, input, train=False):
        pass # returns output vector and caches input

    @abstractmethod
    def backward(self, del_output):
        pass # uses input from last forward() and returns del_input and caches internal gradients

    @abstractmethod
    def update(self, alpha):
        pass # uses cached internal gradients from last backward()

class Linear(Layer):
    def __init__(self, input_size, output_size, W=None, b=None):
        # Kaiming init for ReLU
        self.W = rng.normal(0, np.sqrt(2 / input_size), (input_size, output_size)) if W is None else W
        self.b = np.zeros(output_size) if b is None else b

    def forward(self, input, train=False):
        if train:
            self.input = input
        return input @ self.W + self.b

    def backward(self, del_output):
        assert(self.input.ndim == 2) # TODO support single sample
        batch_size = self.input.shape[0]
        self.dW = (self.input.T @ del_output) / batch_size
        self.db = np.sum(del_output, axis=0) / batch_size
        return del_output @ self.W.T

    def update(self, alpha):
        self.W -= alpha * self.dW
        self.b -= alpha * self.db

class ReLU(Layer):
    def forward(self, input, train=False):
        if train:
            self.input = input
        return np.maximum(0, input)

    def backward(self, del_output):
        return del_output * (self.input >= 0)

    def update(self, alpha):
        pass

class Dropout(Layer):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, input, train=False):
        if train:
            self.mask = (rng.uniform(size=input.shape) >= self.rate) / (1 - self.rate)
            return input * self.mask
        else:
            return input

    def backward(self, del_output):
        return del_output * self.mask

    def update(self, alpha):
        pass

class MLP(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input, train=False):
        output = input
        for l in self.layers:
            output = l.forward(output, train)
        return output

    def backward(self, del_output):
        del_input = del_output
        for l in self.layers[::-1]:
            del_input = l.backward(del_input)
        return del_input

    def update(self, alpha):
        for l in self.layers:
            l.update(alpha)
