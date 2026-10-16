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
    def update(self, cfg):
        """
        cfg is a dict structured as one of the following.
        
        cfg["type"] == "grad"
        - cfg["alpha"]: alpha for gradient descent

        cfg["type"] == "adam"
        - cfg["alpha"]
        - cfg["beta1"]
        - cfg["beta2"]
        - cfg["step"]: 1,2,3,... (for cold start bias correction)
        """
        pass # uses cached internal gradients from last backward()

class Flatten(Layer):
    def forward(self, input, train=False):
        if train:
            self.shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, del_output):
        return del_output.reshape(self.shape)

    def update(self, cfg):
        pass

class Linear(Layer):
    def __init__(self, input_size, output_size, W=None, b=None):
        # Kaiming init for ReLU
        self.W = rng.normal(0, np.sqrt(2 / input_size), (input_size, output_size)) if W is None else W
        self.b = np.zeros(output_size) if b is None else b

        # for Adam optimizer
        self.avg_dW = 0
        self.avg_dW2 = 0
        self.avg_db = 0
        self.avg_db2 = 0

    def forward(self, input, train=False):
        if train:
            self.input = input
        return input @ self.W + self.b

    def backward(self, del_output):
        batch_size = self.input.shape[0]
        self.dW = (self.input.T @ del_output) / batch_size
        self.db = np.sum(del_output, axis=0) / batch_size
        return del_output @ self.W.T

    def update(self, cfg):
        match cfg["type"]:
            case "grad":
                self.W -= cfg["alpha"] * self.dW
                self.b -= cfg["alpha"] * self.db
            case "adam":
                b1, b2, step, alpha = cfg["beta1"], cfg["beta2"], cfg["step"], cfg["alpha"]
                self.avg_dW = b1 * self.avg_dW + (1 - b1) * self.dW
                self.avg_dW2 = b2 * self.avg_dW2 + (1 - b2) * self.dW**2
                self.avg_db = b1 * self.avg_db + (1 - b1) * self.db
                self.avg_db2 = b2 * self.avg_db2 + (1 - b2) * self.db**2

                avg_dW = self.avg_dW / (1 - b1**step)
                avg_dW2 = self.avg_dW2 / (1 - b2**step)
                avg_db = self.avg_db / (1 - b1**step)
                avg_db2 = self.avg_db2 / (1 - b2**step)

                self.W -= alpha * avg_dW / (np.sqrt(avg_dW2) + 1e-10)
                self.b -= alpha * avg_db / (np.sqrt(avg_db2) + 1e-10)
            case _:
                raise Exception(f"{cfg['type']} not supported")

class ReLU(Layer):
    def forward(self, input, train=False):
        if train:
            self.input = input
        return np.maximum(0, input)

    def backward(self, del_output):
        return del_output * (self.input >= 0)

    def update(self, cfg):
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

    def update(self, cfg):
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

    def update(self, cfg):
        for l in self.layers:
            l.update(cfg)
