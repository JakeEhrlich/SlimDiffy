import hypothesis
import hypothesis.strategies as st
import numpy as np
import slimdiffy.autodiff as ad
from function_corpus import all_pairs_tests, dog_elementwise_tests, basic_tensor_tests

@hypothesis.given(st.sampled_from(all_pairs_tests + dog_elementwise_tests + basic_tensor_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    fn_inputs = list(test_obj.arg_strategy.example())
    fn_inputs = [np.array(x) if not isinstance(x, np.ndarray) else x for x in fn_inputs]

    # Get outputs with both direct and interpreted evaluation
    direct_output = test_obj.function(*fn_inputs)
    interpreted_output = ad.jit(test_obj.function)(*fn_inputs)

    # Convert to numpy arrays for comparison
    direct_output = np.array(direct_output)
    interpreted_output = np.array(interpreted_output)

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)
