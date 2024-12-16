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
    shape_strategy = st.lists(sizes, min_size=0, max_size=4).map(tuple)

    # Generate compatible shapes for binary operations
    def generate_compatible_shapes():
        base_shape = shape_strategy.example()
        broadcast_dims = st.integers(min_value=0, max_value=len(base_shape))
        return st.tuples(
            st.just(base_shape),
            broadcast_dims.map(lambda n: base_shape[-n:] if n > 0 else ())
        )

    # Generate arrays with the given shapes
    def make_array(shape):
        return np.random.uniform(-10.0, 10.0, size=shape)

    return st.tuples(
        generate_compatible_shapes(),
        st.floats(min_value=-10.0, max_value=10.0)
    ).map(lambda x: (
        make_array(x[0][0]),
        make_array(x[0][1])
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
    if x.size > 0:
        new_shape = (-1,)
        return np.reshape(x, new_shape)
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
    TestFunc(tensor_corpus_dog_reshape, dog_general_strategy()),
    TestFunc(tensor_corpus_dog_broadcast, dog_general_strategy())
]
