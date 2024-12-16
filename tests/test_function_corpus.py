import numpy as np
import pytest
from hypothesis import given, strategies as st
from .function_corpus import dog_elementwise_tests

@pytest.mark.parametrize("test_func", dog_elementwise_tests)
@given(st.data())
def test_dog_elementwise(test_func, data):
    """Test elementwise operations using dog_general_strategy."""
    args = data.draw(test_func.strategy)
    result = test_func.function(*args)
    assert isinstance(result, np.ndarray)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
