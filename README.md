# SlimDiffy

SlimDiffy is a minimal autodifferentiation library inspired by JAX. It's written in pure Python and uses NumPy directly. It provides a simple yet powerful API for automatic differentiation of Python functions.

## Features

- Automatic differentiation through arbitrary Python functions
- Support for NumPy arrays and operations
- PyTree-like abstraction for structured data
- JIT function decoration
- Basic expression optimization passes

## Installation

```bash
# Not yet published to PyPI
git clone https://github.com/jehrlich/slimdiffy
cd slimdiffy
pip install -e .
```

## Usage

Basic example of taking gradients through a function:

```python
import numpy as np
import slimdiffy.autodiff as ad

@ad.jit
def f(x, y):
    return np.sum(x * y**2)

# Create input arrays
x = np.array([[1., 2.], [3., 4.]])
y = np.array([[5., 6.], [7., 8.]])

# Get gradient function
grad_f = ad.grad(f)
result = grad_f(x, y)

print("dx =", result[0])
print("dy =", result[1])
```

More complex example with nested structures:

```python
from dataclasses import dataclass

@dataclass
class Model:
    weights: np.ndarray
    bias: np.ndarray

@ad.jit
def loss(model, inputs):
    # Compute neural network output
    hidden = inputs @ model.weights + model.bias
    output = ad.sin(hidden)
    return ad.sum(output**2)

# Create model and inputs
model = Model(
    weights=np.array([[1., 2.], [3., 4.]]),
    bias=np.array([0.1, 0.2])
)
inputs = np.array([[0.5, 0.6]])

# Get gradient of loss w.r.t. model params
dmodel, dinputs = ad.grad(loss)(model, inputs)

print("dweights =", dmodel.weights)
print("dbias =", dmodel.bias)
```

## Future Plans

- WebAssembly compilation target
- Numpy operations coverage expansion
- Performance optimizations

## License

MIT
