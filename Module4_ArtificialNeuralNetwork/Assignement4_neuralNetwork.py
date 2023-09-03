import numpy as np

# Define the neural network architecture
input_size = 2
hidden_size = 2
output_size = 2

# Initialize weights and biases randomly
np.random.seed(0)
weights_input_hidden = np.random.rand(input_size, hidden_size)
biases_hidden = np.random.rand(1, hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
biases_output = np.random.rand(1, output_size)

# Define the learning rate and number of iterations
learning_rate = 0.1
num_iterations = 10000

# Define the input data and corresponding target outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0, 0], [0, 1], [0, 1], [1, 1]])

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Training the neural network
for i in range(num_iterations):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = (
        np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    )
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the loss
    loss = 0.5 * np.square(Y - output_layer_output).mean()

    # Backpropagation
    d_output = (Y - output_layer_output) * sigmoid_derivative(output_layer_output)
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(
        hidden_layer_output
    )

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    biases_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if i % 1000 == 0:
        print(f"Iteration {i}: Loss = {loss}")

# Test the trained network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(
    np.dot(
        sigmoid(np.dot(test_input, weights_input_hidden) + biases_hidden),
        weights_hidden_output,
    )
    + biases_output
)
print("Predicted Output:")
print(predicted_output)
