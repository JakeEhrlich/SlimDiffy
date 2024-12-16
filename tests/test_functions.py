import numpy as np
import pytest
from hypothesis import given, strategies as st
import slimdiffy.autodiff as ad
import slimdiffy.pytree as pt
from dataclasses import dataclass

@dataclass
class TestFunc:
    function: callable
    strategy: any

def dog_general_strategy():
    """Generate test cases for dog operations with general tensor shapes."""
    # Start with simple shapes that are guaranteed to be compatible
    simple_shapes = st.one_of(
        st.just((1,)),  # single dimension with size 1 (avoid empty shape for now)
        st.lists(st.just(1), min_size=1, max_size=3).map(tuple),  # shapes with all 1s
        st.integers(min_value=1, max_value=5).map(lambda x: (x,))  # single dimension
    )

    def make_compatible_shapes(shape):
        """Create two shapes that are broadcast-compatible."""
        return st.tuples(st.just(shape), st.just(shape))

    def make_arg_spec(shapes):
        return tuple(ad.ArgSpec(dtype=np.dtype(np.float64), shape=s) for s in shapes)

    return simple_shapes.flatmap(make_compatible_shapes).map(make_arg_spec)

def create_random_tensor(shape):
    """Create a random tensor with given shape."""
    return np.random.uniform(-10.0, 10.0, size=shape).astype(np.float64)

@ad.jit
def tensor_corpus_dog_min(x, y):
    """Element-wise minimum of two tensors."""
    result = x.minimum(y)
    # Ensure result is always an ndarray by adding a new axis
    return np.expand_dims(result, 0) if np.isscalar(result) else result

@ad.jit
def tensor_corpus_dog_max(x, y):
    """Element-wise maximum of two tensors."""
    result = x.maximum(y)
    # Ensure result is always an ndarray by adding a new axis
    return np.expand_dims(result, 0) if np.isscalar(result) else result

@ad.jit
def tensor_corpus_dog_reshape(x):
    """Reshape tensor to a compatible shape."""
    if len(x.shape) > 0:
        size = int(np.prod(x.shape))
        # Pass size as a single integer for 1D reshape
        return x.reshape(size)
    return x.reshape(1)

@ad.jit
def tensor_corpus_dog_broadcast(x, y):
    """Broadcast tensor x to be compatible with shape of y."""
    if len(y.shape) > 0:
        try:
            return x.broadcast_to(y.shape)
        except ValueError:
            x_shape = (1,) * (len(y.shape) - len(x.shape)) + x.shape
            return x.reshape(x_shape).broadcast_to(y.shape)
    return x.reshape(1)

# List of test functions using dog_general_strategy
dog_elementwise_tests = [
    TestFunc(tensor_corpus_dog_min, dog_general_strategy()),
    TestFunc(tensor_corpus_dog_max, dog_general_strategy()),
    TestFunc(tensor_corpus_dog_reshape, dog_general_strategy().map(lambda x: (x[0],))),
    TestFunc(tensor_corpus_dog_broadcast, dog_general_strategy())
]

@pytest.mark.parametrize("test_func", dog_elementwise_tests)
@given(st.data())
def test_dog_elementwise(test_func, data):
    """Test elementwise operations using dog_general_strategy."""
    arg_specs = data.draw(test_func.strategy)

    # Wrap ArgSpec objects in pt.leaf for get_expr
    wrapped_specs = tuple(pt.leaf(spec) for spec in arg_specs)
    expr = test_func.function.get_expr(*wrapped_specs)

    # Create random tensors based on the ArgSpec shapes
    input_tensors = tuple(
        pt.leaf(create_random_tensor(spec.shape))
        for spec in arg_specs
    )

    # Create interpreter and run computation
    interpreter = ad.Interpreter(input_tensors)
    result = interpreter(expr)

    # Verify the result
    assert isinstance(result, np.ndarray)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
