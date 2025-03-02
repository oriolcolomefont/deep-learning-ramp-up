import numpy as np


class LinearRegressionSGD:
    """
    Linear Regression using Stochastic Gradient Descent (SGD) with minibatches.
    """

    def __init__(
        self, learning_rate: float = 0.01, batch_size: int = 32, epochs: int = 2000
    ):
        """
        Initialize the Linear Regression model.

        :param learning_rate: The learning rate for gradient descent.
        :param batch_size: The size of each minibatch for SGD.
        :param epochs: The number of epochs to train the model.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight = np.random.randn()
        self.bias = np.random.randn()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output for given input x.

        :param x: Input features.
        :return: Predicted output.
        """
        return self.weight * x + self.bias

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error (MSE) loss, which measures the average
        squared difference between predicted and actual values. MSE heavily
        penalizes large errors due to the squared term, making it particularly
        sensitive to outliers. The loss is always non-negative, with values
        closer to zero indicating better fit.

        For more details, see:
        https://en.wikipedia.org/wiki/Mean_squared_error

        :param y_true: True output values.
        :param y_pred: Predicted output values.
        :return: Mean squared error loss, computed as mean((y_true - y_pred)^2).
        """
        return np.mean((y_true - y_pred) ** 2)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        """
        Train the model using the provided data.

        :param x: Input features.
        :param y: True output values.
        :param x_val: Validation input features.
        :param y_val: Validation true output values.
        """
        n_samples = x.shape[0]

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred = self.predict(x_batch)

                # Compute gradients using partial derivatives of MSE loss
                # For weight: d(MSE)/dw = -2 * mean(x * (y_true - y_pred))
                # This comes from chain rule: d(MSE)/dw = d(MSE)/d(y_pred)
                # * d(y_pred)/dw
                d_weight = -2 * np.mean(x_batch * (y_batch - y_pred))

                # For bias: d(MSE)/db = -2 * mean(y_true - y_pred)
                # Similarly derived using chain rule: d(MSE)/db = d(MSE)/
                # d(y_pred) * d(y_pred)/db
                d_bias = -2 * np.mean(y_batch - y_pred)

                # Update parameters
                self.weight -= self.learning_rate * d_weight
                self.bias -= self.learning_rate * d_bias

            # Print debug information
            if epoch % 100 == 0:
                train_loss = self.compute_loss(y, self.predict(x))
                if x_val is not None and y_val is not None:
                    val_loss = self.compute_loss(y_val, self.predict(x_val))
                    print(
                        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Weight: {self.weight:.4f}, "
                        f"Bias: {self.bias:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
                        f"Weight: {self.weight:.4f}, Bias: {self.bias:.4f}"
                    )


def generate_synthetic_data(
    n_samples: int = 2000, validation_split: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for training and validation.

    :param n_samples: Number of samples to generate.
    :param validation_split: Fraction of data to be used as validation set.
    :return: Tuple of training and validation input features and target outputs.
    """
    x = np.random.rand(n_samples) * 10
    y = 3.5 * x + np.random.randn(n_samples) * 2

    # Normalize both x and y to zero mean and unit variance
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)

    # Split into training and validation sets
    split_index = int(n_samples * (1 - validation_split))
    x_train, x_val = x[:split_index], x[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    # Generate synthetic data
    x_train, y_train, x_val, y_val = generate_synthetic_data()

    # Initialize and train the model
    model = LinearRegressionSGD(learning_rate=0.005, batch_size=16, epochs=5000)
    model.fit(x_train, y_train, x_val, y_val)

    # Example prediction
    x_new = np.array([5.0])
    y_pred = model.predict(x_new)
    print(f"Prediction for input {x_new[0]}: {y_pred[0]:.2f}")
