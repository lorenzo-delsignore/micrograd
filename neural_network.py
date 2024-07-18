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
        return output if len(output) > 1 else output[0]


class MLP:
    def __init__(self, nin, nouts):
        in_outs = [nin] + nouts
        self.layers = [Layer(in_outs[i], in_outs[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    output = n(x)
    output.backward()
    output.print_graph()
