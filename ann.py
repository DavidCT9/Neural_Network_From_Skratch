from layer import Layer
import math

class ANN:
    def __init__(self, n_inputs, n_outputs, n_hidden, n_per_hidden, alpha):
        self.num_inputs = n_inputs
        self.num_outputs = n_outputs
        self.num_hidden = n_hidden
        self.num_neurons_per_hidden = n_per_hidden
        self.alpha = alpha
        self.layers = []

        if self.num_hidden > 0:
            # Input -> First hidden layer
            self.layers.append(Layer(self.num_neurons_per_hidden, self.num_inputs))
            # Hidden layers
            for _ in range(self.num_hidden - 1):
                self.layers.append(Layer(self.num_neurons_per_hidden, self.num_neurons_per_hidden))
            # Hidden -> Output layer
            self.layers.append(Layer(self.num_outputs, self.num_neurons_per_hidden))
        else:
            # No hidden layer case
            self.layers.append(Layer(self.num_outputs, self.num_inputs))

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, input_values):
        if len(input_values) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(input_values)}")

        inputs = input_values[:]
        outputs = []

        for layer in self.layers:
            outputs = []
            for neuron in layer.neurons:
                neuron.inputs = inputs[:]
                N = sum(w * i for w, i in zip(neuron.weights, inputs))
                N += neuron.bias  # FIXED: was N -= bias
                neuron.output = self.sigmoid(N)
                outputs.append(neuron.output)
            inputs = outputs[:]
        return outputs

    def backward(self, desired_output):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            for j, neuron in enumerate(layer.neurons):
                if i == len(self.layers) - 1:  # Output layer
                    error = desired_output[j] - neuron.output
                    neuron.error_gradient = neuron.output * (1 - neuron.output) * error
                else:
                    # Hidden layer
                    next_layer = self.layers[i + 1]
                    error_grad_sum = sum(next_neuron.error_gradient * next_neuron.weights[j]
                                         for next_neuron in next_layer.neurons)
                    neuron.error_gradient = neuron.output * (1 - neuron.output) * error_grad_sum

        # Update weights and biases
        for i in range(len(self.layers)):
            for neuron in self.layers[i].neurons:
                for k in range(neuron.num_inputs):
                    neuron.weights[k] += self.alpha * neuron.inputs[k] * neuron.error_gradient
                neuron.bias += self.alpha * neuron.error_gradient  # FIXED bias update sign

    def train(self, input_values, desired_output):
        outputs = self.forward(input_values)
        self.backward(desired_output)
        return outputs

    def predict(self, input_values):
        return self.forward(input_values)
