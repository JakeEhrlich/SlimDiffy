import hypothesis
#import hypothesis.strategies as st
import numpy as np
import slimdiffy.autodiff as ad
from function_corpus import get_test_samples, basic_tensor_tests, all_pairs_tests

@hypothesis.given(get_test_samples(basic_tensor_tests + all_pairs_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    func, args, static_argnames = test_obj
    # Get outputs with both direct and interpreted evaluation
    direct_output = func(*args)
    interpreted_output = ad.jit(func, static_argnames=static_argnames)(*args)

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    test_functions()
