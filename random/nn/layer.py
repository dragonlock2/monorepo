import numpy as np
from abc import ABC, abstractmethod

rng = np.random.default_rng(seed=69)

def update(cfg, state, weight, grad):
    match cfg["type"]:
        case "grad":
            weight -= cfg["alpha"] * grad
        case "adam":
            alpha, b1, b2, lamb, step = cfg["alpha"], cfg["beta1"], cfg["beta2"], cfg["lambda"], cfg["step"]
            if state is None:
                state = { "avg": 0, "avg2": 0 }
            state["avg"] = b1 * state["avg"] + (1 - b1) * grad
            state["avg2"] = b2 * state["avg2"] + (1 - b2) * grad**2
            avg = state["avg"] / (1 - b1**step)
            avg2 = state["avg2"] / (1 - b2**step)
            if lamb is not None:
                weight -= alpha * lamb * weight
            weight -= alpha * avg / (np.sqrt(avg2) + 1e-10)
        case _:
            raise Exception(f"{cfg['type']} not supported")
    return weight, state

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
        - cfg["lambda"]: float if apply weight decay, None otherwise
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
        self.dW_state = None
        self.db_state = None

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
        self.W, self.dW_state = update(cfg, self.dW_state, self.W, self.dW)
        self.b, self.db_state = update(cfg, self.db_state, self.b, self.db)

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
