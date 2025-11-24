import random
import math


class Neuron:
    def __init__(self, n_inputs):
        self.num_inputs = n_inputs
        self.bias = random.uniform(-1.0, 1.0)
        self.output = 0.0
        self.error_gradient = 0.0
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.inputs = [0.0 for _ in range(n_inputs)]