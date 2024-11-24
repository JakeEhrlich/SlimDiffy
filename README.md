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
import autodiff as ad

@ad.jit
def f(x, y):
    return np.sum(x * y**2)

# Create input arrays
x = np.array([[1., 2.], [3., 4.]])
y = np.array([[5., 6.], [7., 8.]])

# Get gradient function
grad_f = ad.transform_pipeline(ad.Gradient())
expr = f.get_expr(x, y)
grad = f.transform(grad_f, x, y)

# Can evaluate the gradient with interpreter
interpreter = ad.Interpreter((x, y))
dx, dy = interpreter(grad)

print("dx =", dx)
print("dy =", dy)
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
    return np.sum(output**2)

# Create model and inputs
model = Model(
    weights=np.array([[1., 2.], [3., 4.]]),
    bias=np.array([0.1, 0.2])
)
inputs = np.array([[0.5, 0.6]])

# Get gradient of loss w.r.t. model params
grad_loss = ad.transform_pipeline(
    ad.Gradient(),
    ad.CommonSubexpressionElimination(),
    ad.ConstantFolding(),
    ad.DeadCodeElimination()
)

expr = loss.get_expr(model, inputs)
grad = loss.transform(grad_loss, model, inputs)

interpreter = ad.Interpreter((model, inputs))
dmodel = interpreter(grad)

print("dweights =", dmodel.weights)
print("dbias =", dmodel.bias)
```

## Future Plans

- WebAssembly compilation target
- Numpy operations coverage expansion
- Performance optimizations
- Multi-device support

## License

MIT