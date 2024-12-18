import hypothesis
import hypothesis.strategies as st
import numpy as np
import slimdiffy.autodiff as ad
from slimdiffy.autodiff import grad, Gradient
import slimdiffy.pytree as pt
from function_corpus import (
    get_test_samples, basic_tensor_tests, all_pairs_tests,
    TestFunc, tensor_courpus_add, elementwise_strategy
)

def _to_numpy(x):
    if isinstance(x, pt.Node):
        return _to_numpy(x.leaf_value)
    if isinstance(x, ad.Tracer):
        equation = x.supervisor.equations[x.idx]
        print(f"Converting equation type: {type(equation)}")
        if isinstance(equation, ad.Literal):
            print(f"Found Literal with value type: {type(equation.value)}")
            return equation.value
        if hasattr(equation, 'array'):
            print(f"Found array with shape: {equation.array.shape}")
            return equation.array
        return _to_numpy(equation)
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def compute_finite_diff_jacobian(func, args, eps=1e-7):
    """Compute jacobian using finite differences."""
    print(f"\nComputing finite differences with eps={eps}")
    base_output = func(*args)
    print(f"Base output shape: {base_output.shape if hasattr(base_output, 'shape') else type(base_output)}")

    jac = []
    for i, arg in enumerate(args):
        if not isinstance(arg, np.ndarray):
            print(f"Skipping non-array argument {i}: {type(arg)}")
            continue

        print(f"\nComputing jacobian for arg {i} with shape {arg.shape}")

        # Create a jacobian matrix for this argument
        if not hasattr(base_output, 'shape'):
            print(f"Output is not an array: {type(base_output)}")
            continue

        # Flatten input and create perturbation matrix
        flat_arg = arg.flatten()
        n_elements = flat_arg.size
        print(f"Processing {n_elements} elements for arg {i}")

        # Initialize output array
        jac_i = np.zeros((base_output.size, n_elements))

        # Compute derivatives for each input element
        for j in range(n_elements):
            if j % 100 == 0:  # Progress indicator for large arrays
                print(f"Processing element {j}/{n_elements}")

            # Create perturbation
            pert = np.zeros_like(flat_arg)
            pert[j] = eps
            pert = pert.reshape(arg.shape)

            # Forward difference
            args_plus = list(args)
            args_plus[i] = arg + pert
            output_plus = func(*args_plus)

            # Compute derivative
            deriv = (output_plus - base_output) / eps
            jac_i[:, j] = deriv.flatten()

        # Reshape jacobian to match output and input shapes
        jac_i = jac_i.reshape(base_output.shape + arg.shape)
        jac.append(jac_i)
        print(f"Completed jacobian for arg {i}, shape: {jac_i.shape}")

    return jac

@hypothesis.given(get_test_samples(basic_tensor_tests + all_pairs_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    func, args, static_argnames = test_obj
    # Get outputs with both direct and interpreted evaluation
    direct_output = func(*args)
    interpreted_output = ad.jit(func, static_argnames=static_argnames)(*args)

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)

@hypothesis.given(get_test_samples([
    TestFunc(tensor_courpus_add, elementwise_strategy(2, values=[
        st.just(-1.0),  # Fixed value for first argument
        st.just(1.0)    # Fixed value for second argument
    ]))
]))
def test_jacobians(test_obj):
    """Test that jacobian computation matches finite difference approximation"""
    func, args, static_argnames = test_obj
    print(f"\nTesting function: {func.__name__}")
    print(f"Args shapes: {[arg.shape if hasattr(arg, 'shape') else arg for arg in args]}")
    print(f"Args values: {[arg if not hasattr(arg, 'shape') else 'array' for arg in args]}")

    # Create grad transform with Gradient configured for full jacobian
    grad_transform = grad(func, static_argnames=static_argnames)
    # Override the default gradient_only=True in the first transform (Gradient)
    grad_transform.transforms = (Gradient(gradient_only=False),) + grad_transform.transforms[1:]

    print("\nComputing autodiff jacobian...")
    autodiff_jac = grad_transform(*args)
    print(f"Autodiff jacobian shapes: {[j.shape if hasattr(j, 'shape') else j for j in autodiff_jac]}")

    print("\nComputing finite difference jacobian...")
    finite_diff_jac = compute_finite_diff_jacobian(func, args)
    print(f"Finite diff jacobian shapes: {[j.shape if hasattr(j, 'shape') else j for j in finite_diff_jac]}")

    # Compare results
    print("\nComparing results...")
    for i, (ad_j, fd_j) in enumerate(zip(autodiff_jac, finite_diff_jac)):
        print(f"\nComparing jacobian {i}:")
        print(f"Autodiff shape: {ad_j.shape}, Finite diff shape: {fd_j.shape}")
        print(f"Autodiff type: {type(ad_j)}")
        # Convert autodiff jacobian to numpy array before comparison
        ad_j_np = _to_numpy(ad_j)
        print(f"Converted autodiff jacobian:\n{ad_j_np}")
        print(f"Finite diff jacobian:\n{fd_j}")
        diff = np.abs(ad_j_np - fd_j)
        print(f"Absolute differences:\n{diff}")
        print(f"Max absolute difference: {np.max(diff)}")
        np.testing.assert_allclose(ad_j_np, fd_j, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    test_functions()
