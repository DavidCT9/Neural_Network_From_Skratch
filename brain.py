# brain.py
from ann import ANN

if __name__ == "__main__":
    ann = ANN(2, 1, 1, 2, 0.8)

    # XOR dataset
    training_data = [
        ([1, 1], [0]),
        ([1, 0], [1]),
        ([0, 1], [1]),
        ([0, 0], [0])
    ]

    for epoch in range(10000):
        sse = 0.0
        for inputs, expected in training_data:
            result = ann.train(inputs, expected)
            sse += (expected[0] - result[0]) ** 2
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} SSE: {sse:.6f}")

    print("\nFinal Predictions:")
    for inputs, expected in training_data:
        output = ann.predict(inputs)
        print(f"Input: {inputs}, Predicted: {output[0]:.4f}, Expected: {expected[0]}")
