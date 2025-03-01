import numpy as np
from tqdm import tqdm


class LinearRegressionSGD:
    """
    Linear Regression using Stochastic Gradient Descent (SGD) with minibatches.
    """

    def __init__(
        self, 
        learning_rate: float = 0.01, 
        batch_size: int = 32, 
        epochs: int = 2000
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the provided data.

        :param x: Input features.
        :param y: True output values.
        """
        n_samples = x.shape[0]

        with tqdm(total=self.epochs, desc="Training") as pbar:
            for epoch in range(self.epochs):
                indices = np.random.permutation(n_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]

                for start in range(0, n_samples, self.batch_size):
                    end = start + self.batch_size
                    x_batch = x_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    y_pred = self.predict(x_batch)
                    loss = self.compute_loss(y_batch, y_pred)

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
                    tqdm.write(
                        f"Epoch {epoch}, Loss: {loss:.4f}, "
                        f"Weight: {self.weight:.4f}, Bias: {self.bias:.4f}, "
                        f"dWeight: {d_weight:.4f}, dBias: {d_bias:.4f}"
                    )

                pbar.update(1)


def generate_synthetic_data(
    n_samples: int = 1000
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate synthetic data for training.

    :param n_samples: Number of samples to generate.
    :return: Tuple of input features, target outputs, and normalization
    parameters.
    """
    x = np.random.rand(n_samples) * 10  # Random inputs
    y = 3.5 * x + np.random.randn(n_samples) * 2  # Linear function with noise
    
    # Normalize the input features
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_normalized = (x - x_mean) / x_std
    
    # Normalize the target variable
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    return x_normalized, y_normalized, y_mean, y_std


if __name__ == "__main__":
    # Generate synthetic data
    x_train, y_train, y_mean, y_std = generate_synthetic_data()

    # Initialize and train the model
    model = LinearRegressionSGD(
        learning_rate=0.01, 
        batch_size=32, 
        epochs=2000
    )
    model.fit(x_train, y_train)

    # Example prediction
    x_new = np.array([5.0])
    y_pred_normalized = model.predict(x_new)
    y_pred = y_pred_normalized * y_std + y_mean  # Denormalize the prediction
    print(f"Prediction for input {x_new[0]}: {y_pred[0]:.2f}")
