import layer
import optimize
import numpy as np
from tensorflow.keras.datasets import mnist

"""
MLP from scratch.
"""

class MNIST_MLP(layer.MLP):
    def __init__(self):
        super().__init__(( # TODO add dropout
            layer.Linear(784, 1024),
            layer.ReLU(),
            layer.Dropout(0.1),
            layer.Linear(1024, 512),
            layer.ReLU(),
            layer.Dropout(0.1),
            layer.Linear(512, 256),
            layer.ReLU(),
            layer.Dropout(0.1),
            layer.Linear(256, 10),
        ))

    def predict(self, input):
        if input.ndim == 1:
            return np.argmax(self.forward(input))
        else:
            return np.argmax(self.forward(input), axis=1)

if __name__ == "__main__":
    model = MNIST_MLP()
    (input_train, output_train), (input_test, output_test) = mnist.load_data()
    input_train, input_test = input_train / 255.0, input_test / 255.0
    input_train, input_test = input_train.reshape(input_train.shape[0], -1), input_test.reshape(input_test.shape[0], -1)

    # train
    optimize.BGD(model, input_train, output_train)

    # test
    print(f"train accuracy: {np.sum(model.predict(input_train) == output_train) / len(output_train) :.4f}")
    print(f"test accuracy: {np.sum(model.predict(input_test) == output_test) / len(output_test) :.4f}")
