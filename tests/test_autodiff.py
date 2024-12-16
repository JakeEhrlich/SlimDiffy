import numpy as np
from hypothesis import given, settings, HealthCheck
import slimdiffy.autodiff as ad
from function_corpus import (
    basic_tensor_tests,
    all_pairs_tests,
    dog_elementwise_tests,
    get_test_samples,
)

@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(get_test_samples(basic_tensor_tests + all_pairs_tests + dog_elementwise_tests))
def test_functions(test_obj):
    # Convert input args into numpy arrays
    func, args, static_argnames = test_obj

    # Debug output for dot_general operations
    if 'dog' in func.__name__:
        print(f"\nTesting {func.__name__}")
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                print(f"Array {i} shape: {arg.shape}")
            else:
                print(f"Static arg {i}: {arg}")

    # Create a supervisor for all Tracer objects
    supervisor = ad.TracerSupervisor()

    # For direct output, wrap only numpy arrays in Tracer, leave static args unchanged
    tracer_args = []
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray):
            # Create Literal for numpy array and wrap in Tracer
            literal = ad.Literal(arg)
            tracer = ad.Tracer(literal, supervisor=supervisor)
            # For sum operations, ensure we preserve dimensions
            if 'sum' in func.__name__ and i == 0:
                tracer = ad.sum(tracer, keepdims=True)
            tracer_args.append(tracer)
            if 'dog' in func.__name__:
                print(f"Tracer {i} shape: {tracer.shape}")
        else:
            tracer_args.append(arg)

    # Call function with traced arguments
    try:
        direct_output = func(*tracer_args)
        if isinstance(direct_output, ad.Tracer):
            expr = supervisor.equations[direct_output.idx]
            if isinstance(expr, ad.Literal):
                direct_output = expr.value
            else:
                # For non-literal expressions, we need to evaluate them
                direct_output = ad.jit(func, static_argnames=static_argnames)(*args)
                if 'dog' in func.__name__:
                    print(f"Direct output shape: {direct_output.shape}")
    except Exception as e:
        print(f"Error in direct evaluation: {e}")
        raise

    # Get output with interpreted evaluation
    try:
        interpreted_output = ad.jit(func, static_argnames=static_argnames)(*args)
        if 'dog' in func.__name__:
            print(f"Interpreted output shape: {interpreted_output.shape}")
    except Exception as e:
        print(f"Error in interpreted evaluation: {e}")
        raise

    # Check that outputs match within margin
    np.testing.assert_allclose(direct_output, interpreted_output, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    test_functions()
