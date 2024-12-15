import hypothesis
#import hypothesis.strategies as st
import numpy as np
import slimdiffy.autodiff as ad
from function_corpus import TestFunc, tensor_courpus_add_mul, broadcasted_elementwise_strategy, get_test_samples, all_pairs_tests, dog_elementwise_tests, basic_tensor_tests

basic_test = [TestFunc(tensor_courpus_add_mul, broadcasted_elementwise_strategy(2))]

@hypothesis.given(get_test_samples(basic_tensor_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    func, args = test_obj
    # Get outputs with both direct and interpreted evaluation
    direct_output = func(*args)
    interpreted_output = ad.jit(func)(*args)

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    test_functions()
