import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# Check for NumPy version compatibility
if np.lib.NumpyVersion(np.__version__) >= np.lib.NumpyVersion("2.0.0"):
    print(
        "Warning: This script may not be fully compatible with NumPy 2.x. "
        "Consider downgrading to NumPy < 2.0."
    )


class FeedforwardNN(nn.Module):
    """
    Feedforward Neural Network using Stochastic Gradient Descent (SGD) with minibatches.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 32,
        epochs: int = 2000,
        input_dim: int = 1,
        hidden_dim: int = 10,
    ):
        """
        Initialize the Feedforward Neural Network model.

        :param learning_rate: The learning rate for gradient descent.
        :param batch_size: The size of each minibatch for SGD.
        :param epochs: The number of epochs to train the model.
        :param input_dim: The dimension of the input features.
        :param hidden_dim: The number of neurons in the hidden layer.
        """
        super(FeedforwardNN, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict the output for given input x.

        :param x: Input features.
        :return: Predicted output.
        """
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        """
        Train the model using the provided data.

        :param x_train: Input features for training.
        :param y_train: True output values for training.
        :param x_val: Input features for validation.
        :param y_val: True output values for validation.
        """
        # Convert NumPy arrays to PyTorch tensors with explicit dtype
        x_tensor = torch.tensor(x_train, dtype=torch.float32)  # No need to reshape
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

        train_losses = []

        for epoch in range(self.epochs):
            # Shuffle data
            indices = torch.randperm(x_tensor.size(0))
            x_shuffled = x_tensor[indices]
            y_shuffled = y_tensor[indices]

            # Mini-batch training
            for start in range(0, x_tensor.size(0), self.batch_size):
                end = start + self.batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred = self.forward(x_batch)
                loss = criterion(y_pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Step the scheduler
            scheduler.step()

            # Record training loss
            train_loss = criterion(self.forward(x_tensor), y_tensor).item()
            train_losses.append(train_loss)

            # Print debug information
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
                )

        # Plot the training loss
        plt.plot(range(self.epochs), train_losses, label="Train Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.show()


def generate_synthetic_data(
    n_samples: int = 1000, validation_split: float = 0.2, input_dim: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data for training and validation.

    :param n_samples: Number of samples to generate.
    :param validation_split: Fraction of data to be used as validation set.
    :param input_dim: The dimension of the input features.
    :return: Tuple of training and validation input features and target outputs.
    """
    x = np.random.rand(n_samples, input_dim) * 10
    y = 3.5 * x.sum(axis=1) + np.random.randn(n_samples) * 2

    # Normalize both x and y to zero mean and unit variance
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    # Split into training and validation sets
    split_index = int(n_samples * (1 - validation_split))
    x_train, x_val = x[:split_index], x[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    # Generate synthetic data with input_dim=1 for scalar inputs
    x_train, y_train, x_val, y_val = generate_synthetic_data(input_dim=1)

    # Initialize and train the model with input_dim=1 for scalar inputs
    model = FeedforwardNN(
        learning_rate=0.01, batch_size=32, epochs=1000, input_dim=1, hidden_dim=10
    )
    model.fit(x_train, y_train, x_val, y_val)

    # Example prediction with vector input
    x_new = torch.tensor([[5.0]], dtype=torch.float32)  # Example with single
    y_pred = model.forward(x_new)
    print(f"Prediction for input {x_new.item()}: {y_pred.item():.2f}")
