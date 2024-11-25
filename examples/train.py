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

def get_lr_trap(epoch: int, num_epochs: int) -> float:
    """Get learning rate based on training progress"""
    prog = epoch / num_epochs
    if prog < 0.1:
        return 0.001 + 0.099 * (prog / 0.1)
    elif prog > 0.8:
        return 0.001 + 0.099 * ((1.0 - prog) / 0.1)
    return 0.1

def get_lr_tri(epoch: int, num_epochs: int) -> float:
    """Get learning rate with warmup and cooldown"""
    prog = epoch / num_epochs
    base_lr = 0.001
    peak_lr = 0.1
    warmup_ratio = 0.5
    cooldown_ratio = 0.5

    if prog < warmup_ratio:
        # Linear warmup
        return base_lr + (peak_lr - base_lr) * (prog / warmup_ratio)
    elif prog > (1.0 - cooldown_ratio):
        # Linear cooldown
        cooldown_prog = (prog - (1.0 - cooldown_ratio)) / cooldown_ratio
        return peak_lr + (base_lr - peak_lr) * cooldown_prog
    else:
        # Constant peak learning rate
        return peak_lr

@ad.jit
def loss(model: Model, x, y):
    """MSE loss between model output and targets"""
    pred = forward(model, x)
    loss = ad.sum((pred - y)**2, axis=0)
    return loss / len(x)

def get_training_config(num_params):
    # For small models, we want more optimization steps
    # since each step is computationally cheap
    base_steps = 10_000
    total_steps = int(base_steps * (1 + np.log10(num_params / 1000)))

    # Warmup steps typically 10% of total steps for stability
    warmup_steps = total_steps // 10

    batch_size = int(4 * np.log2(num_params))

    # Learning rate scales with batch size but needs to be
    # larger for small models due to sharper optimization landscape
    base_lr = 0.02
    max_lr = base_lr * np.sqrt(batch_size / 16)
    min_lr = max_lr * 0.02

    def get_lr_schedule(step):
        """Learning rate schedule combining linear warmup and cosine decay"""
        if step < warmup_steps:
            # Linear warmup
            return (step / warmup_steps) * max_lr
        else:
            # Cosine decay with minimum learning rate
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (max_lr - min_lr) * cosine_decay

    return batch_size, total_steps, get_lr_schedule


if __name__ == "__main__":
    # Generate training data
    x_train = np.linspace(-5, 5, 10000)[:, None]
    y_train = test_funcs(x_train)

    # Initialize model and training
    model = init_model(hidden_dim=128)

    num_params = 6 * 128 + 2 * 128 * 128
    batch_size, num_steps, schedule = get_training_config(num_params)

    # Initialize momentum state
    beta = 0.9 # Momentum coefficient
    velocity = pt.map(lambda x: np.zeros_like(x), pt.from_value(model))

    # Training loop
    loss_grad = ad.grad(loss)
    for step in range(num_steps):
        # Random batch
        idx = np.random.randint(0, len(x_train), batch_size)
        x_batch = x_train[idx]
        y_batch = y_train[idx]

        # Get gradients
        model_grad, _, _ = loss_grad(model, x_batch, y_batch)

        # Update with momentum SGD
        lr = schedule(step)
        def momentum_update(p, g, v):
            beta = 0.90 - lr
            v_new = beta * v + (1-beta) * g
            return p - lr * v_new, v_new

        model_grad_tree = pt.from_value(model_grad)
        model_tree = pt.from_value(model)
        updated = pt.mapkeys(lambda _, p, g, v: momentum_update(p, g, v),
                           model_tree, model_grad_tree, velocity)
        model = pt.map(lambda x: x[0], updated).to_value()
        velocity = pt.map(lambda x: x[1], updated)

        if step % 50 == 0:
            train_loss = loss(model, x_train, y_train)
            print(f"Step {step}: loss = {train_loss}, lr = {lr:.4f}")
