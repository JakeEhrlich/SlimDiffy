import slimdiffy.autodiff as ad
import slimdiffy.pytree as pt
import numpy as np
from dataclasses import dataclass

@dataclass
class Layer:
    weights: np.ndarray
    bias: np.ndarray

@dataclass
class Model:
    layers: list[Layer]

def init_model(input_dim=1, hidden_dim=4, output_dim=1, num_hidden_layers=2):
    """Initialize a neural network with configurable number of hidden layers"""
    layers = []

    # Input layer
    layers.append(Layer(
        np.random.normal(0, 0.1, (input_dim, hidden_dim)),
        np.random.normal(0, 0.1, (1, hidden_dim))
    ))

    # Hidden layers
    for _ in range(num_hidden_layers):
        layers.append(Layer(
            np.random.normal(0, 0.1, (hidden_dim, hidden_dim)),
            np.random.normal(0, 0.1, (1, hidden_dim))
        ))

    # Output layer
    layers.append(Layer(
        np.random.normal(0, 0.1, (hidden_dim, output_dim)),
        np.random.normal(0, 0.1, (1, output_dim))
    ))

    return Model(layers)

def forward(model, x):
    """Forward pass through model"""
    hidden = x
    for layer in model.layers[:-1]:
        hidden = hidden @ layer.weights + layer.bias
        hidden = ad.maximum(0, hidden)

    # Output layer (no activation)
    out = hidden @ model.layers[-1].weights + model.layers[-1].bias
    return out

def test_funcs(x):
    """Generate test function outputs for training"""
    return np.sin(2*x) + np.cos(3*x) + np.exp(-0.1*x**2)

def get_lr(epoch: int, num_epochs: int) -> float:
    """Get learning rate based on training progress"""
    prog = epoch / num_epochs
    if prog < 0.1:
        return 0.001 + 0.099 * (prog / 0.1)
    elif prog > 0.9:
        return 0.001 + 0.099 * ((1.0 - prog) / 0.1)
    return 0.1

@ad.jit
def loss(model: Model, x, y):
    """MSE loss between model output and targets"""
    pred = forward(model, x)
    loss = ad.sum((pred - y)**2, axis=0)
    return loss / len(x)

if __name__ == "__main__":
    # Generate training data
    x_train = np.linspace(-5, 5, 10000)[:, None]
    y_train = test_funcs(x_train)

    # Initialize model and training
    model = init_model(hidden_dim=128)
    num_epochs = 10000
    batch_size = 64

    # Training loop
    loss_grad = ad.grad(loss)
    for epoch in range(num_epochs):
        # Random batch
        idx = np.random.randint(0, len(x_train), batch_size)
        x_batch = x_train[idx]
        y_batch = y_train[idx]

        # Get gradients
        model_grad, _, _ = loss_grad(model, x_batch, y_batch)

        # Update with SGD
        lr = get_lr(epoch, num_epochs)
        def param_update(param, grad):
            return param - lr * grad
        model = pt.map(param_update, pt.from_value(model), pt.from_value(model_grad)).to_value()

        if epoch % 50 == 0:
            train_loss = loss(model, x_train, y_train)
            print(f"Epoch {epoch}: loss = {train_loss}, lr = {lr:.4f}")
