import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

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
        input_dim: int = 784,  # 28x28 images
        hidden_dims: list[int] = [128, 64],
    ):
        """
        Initialize the Feedforward Neural Network model.

        :param learning_rate: The learning rate for gradient descent.
        :param batch_size: The size of each minibatch for SGD.
        :param epochs: The number of epochs to train the model.
        :param input_dim: The dimension of the input features.
        :param hidden_dims: The number of neurons in the hidden layers.
        """
        super(FeedforwardNN, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.hidden_layers = nn.ModuleList()

        # Create hidden layers
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim

        self.output_layer = nn.Linear(last_dim, 10)  # 10 classes for MNIST

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict the output for given input x.

        :param x: Input features.
        :return: Predicted output.
        """
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

    def fit(self, train_loader, test_loader) -> None:
        """
        Train the model using the provided data.

        :param train_loader: DataLoader for training data.
        :param test_loader: DataLoader for test data.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

        train_losses = []
        test_losses = []

        for epoch in range(self.epochs):
            self.train()
            epoch_train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view(x_batch.size(0), -1)  # Flatten images
                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            train_losses.append(epoch_train_loss / len(train_loader))

            # Evaluate on test set
            self.eval()
            epoch_test_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.view(x_batch.size(0), -1)
                    y_pred = self.forward(x_batch)
                    loss = criterion(y_pred, y_batch)
                    epoch_test_loss += loss.item()

            test_losses.append(epoch_test_loss / len(test_loader))

            scheduler.step()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, "
                    f"Test Loss: {test_losses[-1]:.4f}"
                )

        # Plot the training and test loss
        plt.plot(range(self.epochs), train_losses, label="Train Loss")
        plt.plot(range(self.epochs), test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Test Loss over Epochs")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Initialize and train the model
    model = FeedforwardNN(
        learning_rate=0.01,
        batch_size=32,
        epochs=20,
        input_dim=784,
        hidden_dims=[128, 64],
    )
    model.fit(train_loader, test_loader)

    # Example prediction with vector input
    x_new = torch.tensor([[5.0]], dtype=torch.float32)  # Example with single
    y_pred = model.forward(x_new)
    print(f"Prediction for input {x_new.item()}: {y_pred.item():.2f}")
