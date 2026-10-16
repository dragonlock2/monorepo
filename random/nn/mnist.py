import numpy as np
import time
from tensorflow.keras.datasets import mnist

import layer
import optimize

"""
MLP from scratch.
"""

class MNIST_MLP(layer.MLP):
    def __init__(self):
        super().__init__((
            # TODO AdamW
            # TODO CNN
            layer.Flatten(),
            layer.Linear(784, 256),
            layer.ReLU(),
            layer.Dropout(0.1),
            layer.Linear(256, 10),
        ))

    def predict(self, input):
        return np.argmax(self.forward(input), axis=1)

if __name__ == "__main__":
    model = MNIST_MLP()
    (input_train, output_train), (input_test, output_test) = mnist.load_data()
    input_train, input_test = input_train / 255.0, input_test / 255.0

    # train
    start = time.time()
    optimize.MBGD(model, input_train, output_train)
    elapse = time.time() - start
    print(f"training took {elapse:.1f}s")

    # test
    print(f"train accuracy: {np.sum(model.predict(input_train) == output_train) / len(output_train) :.4f}")
    print(f"test accuracy: {np.sum(model.predict(input_test) == output_test) / len(output_test) :.4f}")
