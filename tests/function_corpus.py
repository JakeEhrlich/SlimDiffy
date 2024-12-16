import numpy as np
import slimdiffy.autodiff as ad
import slimdiffy.pytree as pt
from dataclasses import dataclass
from hypothesis import strategies as st

@dataclass
class TestFunc:
    function: callable
    strategy: any

def dog_general_strategy():
    """Generate test cases for dog operations with general tensor shapes."""
    # Generate reasonable dimensions for tensor shapes
    sizes = st.integers(min_value=1, max_value=5)
    shapes = st.lists(sizes, min_size=0, max_size=4).map(tuple)

    # Generate compatible shapes for binary operations
    binary_shapes = st.tuples(shapes, shapes)

    # Create ArgSpec for the shapes
    def make_arg_spec(shape):
        return pt.leaf(ad.ArgSpec(np.float64, shape))

    return binary_shapes.map(lambda shapes: (
        make_arg_spec(shapes[0]),
        make_arg_spec(shapes[1])
    ))

@ad.jit
def tensor_corpus_dog_min(x, y):
    """Element-wise minimum of two tensors."""
    return np.minimum(x, y)

@ad.jit
def tensor_corpus_dog_max(x, y):
    """Element-wise maximum of two tensors."""
    return np.maximum(x, y)

@ad.jit
def tensor_corpus_dog_reshape(x):
    """Reshape tensor to a compatible shape."""
    shape = np.shape(x)
    if shape[0] > 0:
        return np.reshape(x, (-1,))
    return x

@ad.jit
def tensor_corpus_dog_broadcast(x, y):
    """Broadcast tensor x to be compatible with shape of y."""
    try:
        return np.broadcast_to(x, np.shape(y))
    except ValueError:
        return x

# List of test functions using dog_general_strategy
dog_elementwise_tests = [
    TestFunc(tensor_corpus_dog_min, dog_general_strategy()),
    TestFunc(tensor_corpus_dog_max, dog_general_strategy()),
    TestFunc(tensor_corpus_dog_reshape, dog_general_strategy().map(lambda x: (x[0],))),
    TestFunc(tensor_corpus_dog_broadcast, dog_general_strategy())
]
