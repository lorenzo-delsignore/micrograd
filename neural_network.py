import random

from backpropagation import Value


class Neuron:
    def __init__(self, nin):
        self.weights = [
            Value(random.uniform(-1, 1), label=f"w_{nin}") for _ in range(nin)
        ]
        self.bias = Value(random.uniform(-1, 1), label="bias")

    def __call__(self, x):
        output = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        output = output.tanh()
        return output


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        output = [n(x) for n in self.neurons]
        return output
