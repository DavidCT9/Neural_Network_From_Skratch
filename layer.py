from neuron import Neuron

class Layer:
    def __init__(self, n_neurons, num_neuron_inputs):
        self.num_neurons = n_neurons
        self.neurons = [Neuron(num_neuron_inputs) for _ in range(n_neurons)]