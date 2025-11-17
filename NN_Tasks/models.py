import numpy as np

class CustomNN:
    def __init__(self):
        self.initialized = False

    # --------------------------
    # Initialization
    # --------------------------
    def initialize(self, num_of_layers, num_neurons, use_bias=True,
                   activation='sigmoid', lr=0.01, random_state=None):

        if random_state is not None:
            np.random.seed(random_state)

        self.num_features = 5
        self.num_classes = 3
        self.use_bias = use_bias
        self.activation_name = activation
        self.lr = lr

        # Ensure list length matches number of layers
        if isinstance(num_neurons, int):
            num_neurons = [num_neurons] * num_of_layers
        if len(num_neurons) != num_of_layers:
            raise ValueError("num_neurons list length must equal num_of_layers")

        # ----------- ACTIVATION FUNCTIONS -----------
        def sigmoid(x):
            
            return np.where(
                x >= 0,
                1 / (1 + np.exp(-x)),
                np.exp(x) / (1 + np.exp(x))
            )

        def sigmoid_derivative(a):
            return a * (1 - a)

        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

        elif activation == 'tanh':
            self.activation = np.tanh
            self.activation_derivative = lambda a: 1 - np.square(a)

        else:
            raise ValueError("Choose 'sigmoid' or 'tanh'.")

        # Layer dimensions
        layer_dims = [self.num_features] + num_neurons + [self.num_classes]

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i], layer_dims[i + 1]) * 0.01
            self.weights.append(w)

            if use_bias:
                self.biases.append(np.zeros((1, layer_dims[i + 1])))
            else:
                self.biases.append(np.zeros((1, layer_dims[i + 1])))

        self.initialized = True

    # --------------------------
    # Forward Pass
    # --------------------------
    def _forward(self, X):
        activations = [X]
        zs = []

        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = self.activation(z)

            zs.append(z)
            activations.append(a)

        # Output (Softmax)
        z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
        exp_scores = np.exp(z_out - np.max(z_out, axis=1, keepdims=True))
        a_out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        zs.append(z_out)
        activations.append(a_out)

        return activations, zs

    # --------------------------
    # Backward Pass
    # --------------------------
    def _backward(self, activations, y_true):
        grads_w = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        m = y_true.shape[0]

        # One-hot encoding
        y_one_hot = np.zeros((m, self.num_classes))
        y_one_hot[np.arange(m), y_true] = 1

        # Output layer delta
        delta = activations[-1] - y_one_hot

        # Output weights
        grads_w[-1] = activations[-2].T @ delta / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Hidden layers
        for i in reversed(range(len(self.weights) - 1)):
            delta = (delta @ self.weights[i + 1].T) * self.activation_derivative(activations[i + 1])
            grads_w[i] = activations[i].T @ delta / m
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / m

        return grads_w, grads_b

    # --------------------------
    # Training
    # --------------------------
    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=100, on_epoch=None):

        losses, train_accs, test_accs = [], [], []

        for epoch in range(epochs):

            # Forward pass
            activations, zs = self._forward(X_train)
            y_pred = activations[-1]

            # Cross entropy loss
            m = y_train.shape[0]
            log_likelihood = -np.log(y_pred[np.arange(m), y_train] + 1e-9)
            loss = np.mean(log_likelihood)
            losses.append(loss)

            # Training accuracy
            preds = np.argmax(y_pred, axis=1)
            train_acc = np.mean(preds == y_train)
            train_accs.append(train_acc)

            # Test accuracy each epoch
            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                test_acc = np.mean(test_pred == y_test)
                test_accs.append(test_acc)
            else:
                test_accs.append(None)

            # Backprop
            grads_w, grads_b = self._backward(activations, y_train)

            # Update
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]

            if on_epoch:
                on_epoch(epoch, loss, train_acc)

        return {
            "loss": losses,
            "train_acc": train_accs,
            "test_acc": test_accs,
        }

    # --------------------------
    # Prediction
    # --------------------------
    def predict(self, X):
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)
