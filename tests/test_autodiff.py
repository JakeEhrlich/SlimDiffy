import hypothesis
#import hypothesis.strategies as st
import numpy as np
import slimdiffy.autodiff as ad
from function_corpus import get_test_samples, basic_tensor_tests, all_pairs_tests, dog_elementwise_tests

@hypothesis.settings(max_examples=1000)
@hypothesis.given(get_test_samples(basic_tensor_tests + all_pairs_tests + dog_elementwise_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    func, args, static_argnames = test_obj
    # Get outputs with both direct and interpreted evaluation
    direct_output = func(*args)
    interpreted_output = ad.jit(func, static_argnames=static_argnames)(*args)

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)

@hypothesis.settings(max_examples=1000)
@hypothesis.given(get_test_samples(basic_tensor_tests))
def test_jacobians(test_obj):
    # Unpack test object
    func, args, static_argnames = test_obj

    jit_func = ad.jit(func, static_argnames=static_argnames)
    grad_func = ad.grad(func, static_argnames=static_argnames, wrt=0)

    # Get output shapes to determine if gradients are valid for this test
    try:
        test_out = jit_func(*args)
        np.asarray(test_out)
    except Exception:
        # Skip if output can't be converted to array (ex: tuples)
        return

    # Get analytical jacobian
    jac_analytical = grad_func(*args)

    # Compute numerical jacobian
    h = 1e-8
    x = np.asarray(args[0])
    base = x.flatten()
    n_in = len(base)
    fdiff_grad = []
    for i in range(n_in):
        # Create perturbation vectors
        pos_pert = base.copy()
        neg_pert = base.copy()
        pos_pert[i] += h
        neg_pert[i] -= h

        # Reshape back to original shape
        x_pos = pos_pert.reshape(x.shape)
        x_neg = neg_pert.reshape(x.shape)

        # Forward eval with perturbations
        new_args = (x_pos,) + args[1:]
        pos_out = jit_func(*new_args)
        new_args = (x_neg,) + args[1:]
        neg_out = jit_func(*new_args)

        # Central difference
        fdiff = (pos_out - neg_out) / (2*h)
        fdiff_grad.append(fdiff.flatten())

    jac_numerical = np.vstack(fdiff_grad).T

    # Compare with analytical jacobian
    jac_reshaped = np.asarray(jac_analytical).reshape(jac_numerical.shape)
    np.testing.assert_allclose(jac_reshaped, jac_numerical, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    test_functions()
