import numpy as np
from dataclasses import dataclass
import dataclasses
from typing import Any, Callable
import slimdiffy.autodiff as ad
import hypothesis
import hypothesis.strategies as st
from hypothesis.extra.numpy import arrays

@dataclass
class TestFunc:
    function: Callable
    arg_strategy: Any
    static_argnames: set = dataclasses.field(default_factory=lambda: set())

def shape_strategy():
    # Generate reasonable dimensions for tensor shapes
    sizes = st.integers(min_value=1, max_value=5)
    return st.lists(sizes, min_size=0, max_size=4).map(tuple)

def elementwise_strategy(k: int, values=None):
    # Generate shared shape and arrays with same shape
    base = st.shared(shape_strategy())

    @st.composite
    def generate_tensors(draw):
        # Default strategy if none provided
        if values is None:
            default_values = st.floats(
                allow_infinity=False,
                allow_nan=False,
                min_value=-10.0,
                max_value=10.0
            )
            tensor_values = [default_values] * k
        elif isinstance(values, list):
            assert len(values) == k, f"Expected {k} strategies, got {len(values)}"
            tensor_values = values
        else:
            tensor_values = [values] * k

        shape = draw(base)
        tensors = []
        for i in range(k):
            tensor = draw(arrays(
                np.dtype('float64'),
                shape=shape,
                elements=tensor_values[i]
            ))
            tensors.append(tensor)
        return tuple(tensors)

    return generate_tensors()

@st.composite
def dog_general_strategy(draw):
    """Generate shapes for dot_general with broadcasting after the operation"""
    # Draw base shapes with at least one dimension
    batch_shape = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=4))

    # Generate contracting dimensions that will be the same for both sides
    n_contract = draw(st.integers(min_value=1, max_value=4))
    contract_dims = [draw(st.integers(min_value=1, max_value=5)) for _ in range(n_contract)]
    contract_shape = contract_dims

    # Generate free dimensions for each side
    lhs_free = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=4))
    rhs_free = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=4))

    # Build full shapes in natural order: batch, contract, free
    lhs_shape = tuple(batch_shape + contract_shape + lhs_free)
    rhs_shape = tuple(batch_shape + contract_shape + rhs_free)

    # Map dimension indices in natural order
    n_batch = len(batch_shape)
    n_contract = len(contract_shape)

    # Batch dimensions are at the start
    lhs_batch = tuple(range(n_batch))
    rhs_batch = tuple(range(n_batch))

    # Contract dimensions follow batch dimensions
    lhs_contract = tuple(range(n_batch, n_batch + n_contract))
    rhs_contract = tuple(range(n_batch, n_batch + n_contract))

    # Debug output
    print("\nDOG Strategy Debug:")
    print(f"batch_shape: {batch_shape}")
    print(f"contract_shape: {contract_shape}")
    print(f"lhs_free: {lhs_free}")
    print(f"rhs_free: {rhs_free}")
    print(f"lhs_shape: {lhs_shape}")
    print(f"rhs_shape: {rhs_shape}")
    print(f"lhs_batch: {lhs_batch}")
    print(f"rhs_batch: {rhs_batch}")
    print(f"lhs_contract: {lhs_contract}")
    print(f"rhs_contract: {rhs_contract}")

    # Generate random arrays with these shapes
    lhs_array = draw(arrays(
        np.dtype('float64'),
        shape=lhs_shape,
        elements=st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=-10.0,
            max_value=10.0
        )
    ))

    rhs_array = draw(arrays(
        np.dtype('float64'),
        shape=rhs_shape,
        elements=st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=-10.0,
            max_value=10.0
        )
    ))

    return (
        lhs_array,
        rhs_array,
        lhs_contract,
        rhs_contract,
        lhs_batch,
        rhs_batch
    )

def matmul_strategy():
    # Generate random shapes for the matmul inputs
    sizes = st.integers(min_value=1, max_value=5)
    k = st.shared(sizes) # Shared inner dimension
    m = sizes
    n = sizes

    def generate_arrays(dims):
        m_val, k_val, n_val = dims
        return st.tuples(
            arrays(
                np.dtype('float64'),
                shape=(m_val, k_val),
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            ),
            arrays(
                np.dtype('float64'),
                shape=(k_val, n_val),
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            )
        )

    return st.tuples(m, k, n).flatmap(generate_arrays)

def transpose_strategy():
    # Generate base shape
    base_shape = st.shared(shape_strategy())

    def generate_axes(shape):
        n_dims = len(shape)
        if n_dims == 0:
            return st.just(())
        # Generate permutations of the dimension indices
        return st.permutations(range(n_dims)).map(tuple) | st.just(())

    # Generate array and matching permutation
    def combine_shape_axes(shape):
        return st.tuples(
            arrays(
                np.dtype('float64'),
                shape=shape,
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            ),
            generate_axes(shape)
        )

    return base_shape.flatmap(combine_shape_axes)

@st.composite
def reshape_strategy(draw):
    prod = draw(st.integers(min_value=1, max_value=50))
    factors = []
    for f in [2, 3, 4, 5]:
        while prod % f:
            factors.append(f)
            prod = prod // f

    n_splits1 = draw(st.integers(min_value=0, max_value=len(factors)-1))
    n_splits2 = draw(st.integers(min_value=0, max_value=len(factors)-1))

    rng = draw(st.randoms())
    split_points1 = sorted(rng.sample(range(len(factors)), n_splits1))
    split_points2 = sorted(rng.sample(range(len(factors)), n_splits2))

    #print("split points 1: ", split_points1)
    #print("split points 2: ", split_points2)

    def split(lst, split_points):
        start = 0
        for end in split_points:
            yield int(np.prod(lst[start:end], dtype=np.int64))
            start = end
        if len(lst[start:]) != 0:
            yield int(np.prod(lst[start:], dtype=np.int64))

    f1 = rng.sample(factors, len(factors))
    f2 = rng.sample(factors, len(factors))

    #print("factors: ", factors)
    #print("f1: ", f1)
    #print("f2: ", f2)

    shape1 = tuple(split(f1, split_points1))
    shape2 = tuple(split(f2, split_points2))

    #print("shape1: ", shape1)
    #print("shape2: ", shape2)

    arr = draw(arrays(
        np.dtype('float64'),
        shape=shape1,
        elements=st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=-10.0,
            max_value=10.0
        )
    ))

    return (arr, shape2)

def broadcast_strategy():
    @st.composite
    def inner_strategy(draw):
        # Generate target shape
        target = draw(shape_strategy())

        # If target is empty tuple, source must be empty too
        if len(target) == 0:
            return (draw(arrays(
                np.dtype('float64'),
                shape=(),
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            )), target)

        # For each dimension, we can either:
        # 1. Keep the original size
        # 2. Replace with size 1 for broadcasting
        source_dims = []
        for i, size in enumerate(target):
            choice = draw(st.one_of(
                st.just(size),
                st.just(1)
            ))
            source_dims.append(choice)

        # Generate length to slice dimensions to
        length = draw(st.integers(min_value=0, max_value=len(target)))

        # Create final shape by dropping leading dims
        final_shape = tuple(source_dims[length:])

        # Generate source array with this shape
        source = draw(arrays(
            np.dtype('float64'),
            shape=final_shape,
            elements=st.floats(
                allow_infinity=False,
                allow_nan=False,
                min_value=-10.0,
                max_value=10.0
            )
        ))

        return (source, target)

    return inner_strategy()

def non_zero_float_strategy(min_value=-10.0, max_value=10.0, epsilon=1e-3):
    @st.composite
    def strategy(draw):
        x = draw(st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=min_value,
            max_value=max_value
        ))
        # Ensure minimum absolute value while preserving sign
        return x + epsilon if x >= 0 else -epsilon
    return strategy()

def positive_float_strategy(min_value=0.1, max_value=10.0):
    return st.floats(
        allow_infinity=False,
        allow_nan=False,
        min_value=min_value,
        max_value=max_value
    )

def broadcasted_elementwise_strategy(k: int, values=None):
    """Generate k-ary broadcasting strategy with configurable value ranges.

    Args:
        k: Number of input tensors
        values: Either a single strategy or list of k strategies for tensor values.
               If None, uses default range [-10, 10].
    """
    @st.composite
    def build_tensors(draw):
        # Default strategy if none provided
        if values is None:
            default_values = st.floats(
                allow_infinity=False,
                allow_nan=False,
                min_value=-10.0,
                max_value=10.0
            )
            tensor_values = [default_values] * k
        elif isinstance(values, list):
            assert len(values) == k, f"Expected {k} strategies, got {len(values)}"
            tensor_values = values
        else:
            tensor_values = [values] * k

        base = st.shared(shape_strategy())
        shape = draw(base)
        num_dims = len(shape)

        # Return empty tensors if no dimensions
        if num_dims == 0:
            return draw(elementwise_strategy(k, values))

        # Generate valid "allow change" matrix - at least one unchanged dim per column
        def valid_col():
            return st.lists(st.booleans(), min_size=k, max_size=k).filter(
                lambda col: not all(col)
            )
        allow_changes = [draw(valid_col()) for _ in range(num_dims)]
        allow_changes = list(zip(*allow_changes))  # Transpose to per-tensor masks

        # Pick one tensor to maintain dimension count
        fixed_dims_idx = draw(st.integers(min_value=0, max_value=k-1))

        tensors = []
        for i in range(k):
            # Get dimensions for this tensor
            tensor_shape = list(shape)
            if i != fixed_dims_idx:
                # Get max dims we can remove without affecting unchangeable dims
                max_removable = 0
                for d in range(num_dims):
                    if not allow_changes[i][d]:
                        break
                    max_removable = d + 1

                # Maybe truncate leading dimensions
                if max_removable > 0:
                    n_dims = draw(st.integers(min_value=0, max_value=max_removable))
                    tensor_shape = tensor_shape[n_dims:]
                    allow_changes[i] = allow_changes[i][n_dims:]

            # Randomly change allowed dimensions to 1
            for j, can_change in enumerate(allow_changes[i][:len(tensor_shape)]):
                if can_change and draw(st.booleans()):
                    tensor_shape[j] = 1

            # Generate array with modified shape
            tensors.append(draw(
                arrays(
                    np.dtype('float64'),
                    shape=tuple(tensor_shape),
                    elements=tensor_values[i]
                )
            ))

        return tuple(tensors)

    return build_tensors()  #type: ignore

def generate_broadcast_shape_strategy(base_shape):
    def make_broadcast_shape():
        # Maybe add some leading dimensions
        n_extra = np.random.randint(3)  # 0-2 extra dims
        leading_dims = tuple(np.random.randint(1, 4) for _ in range(n_extra))

        # For remaining dims, either match output dim or use size 1
        broadcast_dims = []
        for dim in base_shape:
            if np.random.random() < 0.5:
                broadcast_dims.append(dim)
            else:
                broadcast_dims.append(1)

        return leading_dims + tuple(broadcast_dims)

    return st.builds(make_broadcast_shape)

@st.composite
def dog_post_brodacast_strategy(draw):
    """Generate shapes for dot_general with broadcasting after the operation"""
    # Get dot general inputs first
    base_strategy = dog_general_strategy()
    lhs_array, rhs_array, lhs_c, rhs_c, lhs_b, rhs_b = draw(base_strategy)

    # Calculate output shape from dot_general
    batch_shape = tuple(lhs_array.shape[i] for i in lhs_b)
    lhs_free = tuple(i for i in range(len(lhs_array.shape)) if i not in lhs_c and i not in lhs_b)
    rhs_free = tuple(i for i in range(len(rhs_array.shape)) if i not in rhs_c and i not in rhs_b)

    # Output shape will be batch_shape + lhs_free_shape + rhs_free_shape
    lhs_free_shape = tuple(lhs_array.shape[i] for i in lhs_free)
    rhs_free_shape = tuple(rhs_array.shape[i] for i in rhs_free)
    dog_output_shape = batch_shape + lhs_free_shape + rhs_free_shape

    print(f"dog_output_shape: {dog_output_shape}")

    # For scalar outputs, we need special handling
    if len(dog_output_shape) == 0:
        # For scalar output, we can either keep it scalar or broadcast it
        z_shape = draw(st.one_of(
            st.just(()),  # Keep scalar
            st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3).map(tuple)  # Broadcast
        ))
    else:
        # Generate broadcast shape for z that's compatible with dog_output_shape
        z_shape = draw(generate_broadcast_shape_strategy(dog_output_shape))

    print(f"z_shape: {z_shape}")

    # Generate the actual arrays
    lhs = draw(arrays(
        np.dtype('float64'),
        shape=lhs_array.shape,
        elements=st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0)
    ))
    rhs = draw(arrays(
        np.dtype('float64'),
        shape=rhs_array.shape,
        elements=st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0)
    ))
    z = draw(arrays(
        np.dtype('float64'),
        shape=z_shape,
        elements=st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0)
    ))

    # Debug output
    print("\nDOG Strategy Debug:")
    print(f"batch_shape: {list(batch_shape)}")
    print(f"lhs_free: {list(lhs_free)}")
    print(f"rhs_free: {list(rhs_free)}")
    print(f"lhs_shape: {lhs.shape}")
    print(f"rhs_shape: {rhs.shape}")
    print(f"lhs_batch: {lhs_b}")
    print(f"rhs_batch: {rhs_b}")
    print(f"lhs_contract: {lhs_c}")
    print(f"rhs_contract: {rhs_c}")
    print(f"dog_output_shape: {dog_output_shape}")
    print(f"z_shape: {z.shape}")

    return lhs, rhs, lhs_c, rhs_c, lhs_b, rhs_b, z


@st.composite
def dog_pre_broadcast_left_strategy(draw):
    """Generate shapes for dot_general with broadcasting before the operation on left input"""
    # Get dot general inputs first
    base_strategy = dog_general_strategy()
    x_orig, y_orig, c1, c2, b1, b2 = draw(base_strategy)
    x = x_orig if isinstance(x_orig, np.ndarray) else np.array(x_orig)

    # Generate broadcast shape for x that's compatible with original shape
    if len(x.shape) == 0:
        final_shape = ()  # scalar shape for empty input
    else:
        # Generate a shape that's compatible for broadcasting
        final_shape = draw(generate_broadcast_shape_strategy(x.shape))
        # Verify shapes are compatible
        try:
            np.broadcast_shapes(x.shape, final_shape)
        except ValueError:
            # If incompatible, use the original shape
            final_shape = x.shape

    # Generate the broadcasted array
    x_broadcast = draw(arrays(
        np.dtype('float64'),
        shape=final_shape,
        elements=st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=-10.0,
            max_value=10.0
        )
    ))

    return x_broadcast, y_orig, c1, c2, b1, b2

@st.composite
def dog_pre_broadcast_right_strategy(draw):
    """Generate shapes for dot_general with broadcasting before the operation on right input"""
    # Get dot general inputs first
    base_strategy = dog_general_strategy()
    x_orig, y_orig, c1, c2, b1, b2 = draw(base_strategy)
    y = y_orig if isinstance(y_orig, np.ndarray) else np.array(y_orig)

    # Generate broadcast shape for y that's compatible with original shape
    if len(y.shape) == 0:
        final_shape = ()  # scalar shape for empty input
    else:
        # Generate a shape that's compatible for broadcasting
        final_shape = draw(generate_broadcast_shape_strategy(y.shape))
        # Verify shapes are compatible
        try:
            np.broadcast_shapes(y.shape, final_shape)
        except ValueError:
            # If incompatible, use the original shape
            final_shape = y.shape

    # Generate the broadcasted array
    y_broadcast = draw(arrays(
        np.dtype('float64'),
        shape=final_shape,
        elements=st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=-10.0,
            max_value=10.0
        )
    ))

    return x_orig, y_broadcast, c1, c2, b1, b2

def tensor_courpus_dog_add(x, y, lhs_c, rhs_c, lhs_b, rhs_b, z):
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b) + z

def tensor_courpus_dog_sub(x, y, lhs_c, rhs_c, lhs_b, rhs_b, z):
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b) - z

def tensor_courpus_dog_mul(x, y, lhs_c, rhs_c, lhs_b, rhs_b, z):
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b) * z

def tensor_courpus_dog_div(x, y, lhs_c, rhs_c, lhs_b, rhs_b, z):
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b) / z

def tensor_courpus_dog_pow(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b) ** 2

def tensor_courpus_dog_neg(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return -x.dot_general(y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b)

def tensor_courpus_dog_exp(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.exp(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_dog_log(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    # Ensure inputs are positive for log operation
    x = np.abs(x) + 1.0  # Make strictly positive
    y = np.abs(y) + 1.0  # Make strictly positive
    return ad.log(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_dog_sin(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.sin(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_dog_cos(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.cos(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_dog_abs(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.abs(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_dog_sum(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.sum(x.dot_general(y, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b))

def tensor_courpus_add_dog(x, y, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (x + y).dot_general(z, lhs_contracting_dims=lhs_c,
                           rhs_contracting_dims=rhs_c,
                           lhs_batch_dims=lhs_b,
                           rhs_batch_dims=rhs_b)

def tensor_courpus_sub_dog(x, y, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (x - y).dot_general(z, lhs_contracting_dims=lhs_c,
                           rhs_contracting_dims=rhs_c,
                           lhs_batch_dims=lhs_b,
                           rhs_batch_dims=rhs_b)

def tensor_courpus_mul_dog(x, y, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (x * y).dot_general(z, lhs_contracting_dims=lhs_c,
                           rhs_contracting_dims=rhs_c,
                           lhs_batch_dims=lhs_b,
                           rhs_batch_dims=rhs_b)

def tensor_courpus_div_dog(x, y, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (x / y).dot_general(z, lhs_contracting_dims=lhs_c,
                           rhs_contracting_dims=rhs_c,
                           lhs_batch_dims=lhs_b,
                           rhs_batch_dims=rhs_b)

def tensor_courpus_pow_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (x ** 2).dot_general(z, lhs_contracting_dims=lhs_c,
                           rhs_contracting_dims=rhs_c,
                           lhs_batch_dims=lhs_b,
                           rhs_batch_dims=rhs_b)

def tensor_courpus_neg_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return (-x).dot_general(z, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b)

def tensor_courpus_exp_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.exp(x).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_log_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.log(x).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_sin_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.sin(x).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_cos_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.cos(x).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_abs_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.abs(x).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_sum_dog(x, z, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.sum(x, keepdims=True).dot_general(z, lhs_contracting_dims=lhs_c,
                             rhs_contracting_dims=rhs_c,
                             lhs_batch_dims=lhs_b,
                             rhs_batch_dims=rhs_b)

def tensor_courpus_add(x, y):
    return x + y

def tensor_courpus_sub(x, y):
    return x - y

def tensor_courpus_mul(x, y):
    return x * y

def tensor_courpus_div(x, y):
    return x / y

def tensor_courpus_pow(x):
    return x ** 2

def tensor_courpus_neg(x):
    return -x

def tensor_courpus_dot(x, y):
    return x @ y

def tensor_courpus_dot_general(x, y, lhs_c, rhs_c, lhs_b, rhs_b):
    return ad.dot_general(x, y, lhs_contracting_dims=lhs_c,
                        rhs_contracting_dims=rhs_c,
                        lhs_batch_dims=lhs_b,
                        rhs_batch_dims=rhs_b)

def tensor_courpus_transpose(x, axes):
    return x.transpose(*axes)

def tensor_courpus_exp(x):
    return ad.exp(x)

def tensor_courpus_log(x):
    return ad.log(x)

def tensor_courpus_sin(x):
    return ad.sin(x)

def tensor_courpus_cos(x):
    return ad.cos(x)

def tensor_courpus_abs(x):
    return ad.abs(x)

def tensor_courpus_sum(x):
    return ad.sum(x)

def tensor_courpus_max(x):
    return ad.max(x)

def tensor_courpus_min(x):
    return ad.min(x)

def tensor_courpus_reshape(x, shape):
    return ad.reshape(x, shape)

def tensor_courpus_broadcast(x, shape):
    return ad.broadcast_to(x, shape)

def tensor_courpus_add_mul(x, y):
    return (x + y) * (x + y)

def tensor_courpus_add_div(x, y):
    return (x + y) / y

def tensor_courpus_add_pow(x, y):
    return (x + y) ** 2

def tensor_courpus_add_exp(x, y):
    return ad.exp(x + y)

def tensor_courpus_add_log(x, y):
    return x + ad.log(y)

def tensor_courpus_add_sin(x, y):
    return ad.sin(x + y)

def tensor_courpus_add_cos(x, y):
    return ad.cos(x + y)

def tensor_courpus_sub_mul(x, y):
    return (x - y) * (x - y)

def tensor_courpus_sub_div(x, y):
    return (x - y) / y

def tensor_courpus_sub_pow(x, y):
    return (x - y) ** 2

def tensor_courpus_sub_exp(x, y):
    return ad.exp(x - y)

def tensor_courpus_sub_log(x, y):
    return x - ad.log(y)

def tensor_courpus_sub_sin(x, y):
    return ad.sin(x - y)

def tensor_courpus_sub_cos(x, y):
    return ad.cos(x - y)

def tensor_courpus_mul_div(x, y):
    return (x * y) / y

def tensor_courpus_mul_pow(x, y):
    return (x * y) ** 2

def tensor_courpus_mul_exp(x, y):
    return ad.exp(x * y)

def tensor_courpus_mul_log(x, y):
    return x * ad.log(y)

def tensor_courpus_mul_sin(x, y):
    return ad.sin(x * y)

def tensor_courpus_mul_cos(x, y):
    return ad.cos(x * y)

def tensor_courpus_sin_cos(x):
    return ad.sin(ad.cos(x))

def tensor_courpus_cos_sin(x):
    return ad.cos(ad.sin(x))

def tensor_courpus_exp_log(x):
    return ad.exp(ad.log(x))

def tensor_courpus_log_exp(x):
    return ad.log(ad.exp(x))

def tensor_courpus_add_min(x, y):
    return ad.minimum(x, y) + ad.minimum(x, y)

def tensor_courpus_add_max(x, y):
    return ad.maximum(x, y) + ad.maximum(x, y)

def tensor_courpus_sub_min(x, y):
    return ad.minimum(x, y) - ad.minimum(x, y)

def tensor_courpus_sub_max(x, y):
    return ad.maximum(x, y) - ad.maximum(x, y)

def tensor_courpus_mul_min(x, y):
    return ad.minimum(x, y) * ad.minimum(x, y)

def tensor_courpus_mul_max(x, y):
    return ad.maximum(x, y) * ad.maximum(x, y)

def tensor_courpus_min_max(x, y):
    return ad.minimum(x, ad.maximum(x, y))

def tensor_courpus_max_min(x, y):
    return ad.maximum(x, ad.minimum(x, y))

def tensor_courpus_min_exp(x, y):
    return ad.minimum(x, ad.exp(y))

def tensor_courpus_min_log(x, y):
    return ad.minimum(x, ad.log(y))

def tensor_courpus_min_sin(x, y):
    return ad.minimum(x, ad.sin(y))

def tensor_courpus_min_cos(x, y):
    return ad.minimum(x, ad.cos(y))

def tensor_courpus_max_exp(x, y):
    return ad.maximum(x, ad.exp(y))

def tensor_courpus_max_log(x, y):
    return ad.maximum(x, ad.log(y))

def tensor_courpus_max_sin(x, y):
    return ad.maximum(x, ad.sin(y))

def tensor_courpus_max_cos(x, y):
    return ad.maximum(x, ad.cos(y))

# Define test lists for basic tensor operations
basic_tensor_tests = [
    TestFunc(tensor_courpus_add, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub, elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul, elementwise_strategy(2)),
    TestFunc(tensor_courpus_div, elementwise_strategy(2)),
    TestFunc(tensor_courpus_pow, elementwise_strategy(1)),
    TestFunc(tensor_courpus_neg, elementwise_strategy(1)),
    TestFunc(tensor_courpus_exp, elementwise_strategy(1)),
    TestFunc(tensor_courpus_log, elementwise_strategy(1, st.floats(min_value=0.1, max_value=10.0))),
    TestFunc(tensor_courpus_sin, elementwise_strategy(1)),
    TestFunc(tensor_courpus_cos, elementwise_strategy(1)),
    TestFunc(tensor_courpus_abs, elementwise_strategy(1)),
    TestFunc(tensor_courpus_sum, elementwise_strategy(1)),
]

# Define test lists for composite operations
all_pairs_tests = [
    TestFunc(tensor_courpus_add_mul, elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_div, elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_pow, elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_exp, elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_log, elementwise_strategy(2, st.floats(min_value=0.1, max_value=10.0))),
    TestFunc(tensor_courpus_add_sin, elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_cos, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_mul, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_div, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_pow, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_exp, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_log, elementwise_strategy(2, st.floats(min_value=0.1, max_value=10.0))),
    TestFunc(tensor_courpus_sub_sin, elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_cos, elementwise_strategy(2)),
]

dog_elementwise_tests = [
    TestFunc(tensor_courpus_dog_add, dog_post_brodacast_strategy()),
    TestFunc(tensor_courpus_dog_sub, dog_post_brodacast_strategy()),
    TestFunc(tensor_courpus_dog_mul, dog_post_brodacast_strategy()),
    TestFunc(tensor_courpus_dog_div, dog_post_brodacast_strategy()),
    TestFunc(tensor_courpus_dog_pow, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_neg, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_exp, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_log, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_sin, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_cos, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_abs, dog_general_strategy()),
    TestFunc(tensor_courpus_dog_sum, dog_general_strategy()),
    TestFunc(tensor_courpus_add_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_sub_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_mul_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_div_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_pow_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_neg_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_exp_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_log_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_sin_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_cos_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_abs_dog, dog_pre_broadcast_left_strategy()),
    TestFunc(tensor_courpus_sum_dog, dog_pre_broadcast_left_strategy())
]


all_pairs_tests = [
    TestFunc(tensor_courpus_add_mul, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_div, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        non_zero_float_strategy(min_value=-10.0, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_add_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_log, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        positive_float_strategy(min_value=0.1, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_add_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_cos, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_min, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_mul, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_div, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        non_zero_float_strategy(min_value=-10.0, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_sub_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_log, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        positive_float_strategy(min_value=0.1, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_sub_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_cos, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_min, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_div, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        non_zero_float_strategy(min_value=-10.0, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_mul_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_log, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        positive_float_strategy(min_value=0.1, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_mul_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_cos, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_min, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sin_cos, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_cos_sin, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_exp_log, broadcasted_elementwise_strategy(1, values=
         positive_float_strategy(min_value=0.1, max_value=10.0)
    )),
    TestFunc(tensor_courpus_log_exp, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_min_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_max_min, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_min_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_min_log, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        positive_float_strategy(min_value=0.1, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_min_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_min_cos, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_max_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_max_log, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        positive_float_strategy(min_value=0.1, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_max_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_max_cos, broadcasted_elementwise_strategy(2)),
]

basic_tensor_tests = [
    TestFunc(tensor_courpus_add, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_div, broadcasted_elementwise_strategy(2, values=[
        st.floats(allow_infinity=False, allow_nan=False, min_value=-10.0, max_value=10.0),
        non_zero_float_strategy(min_value=-10.0, max_value=10.0)
    ])),
    TestFunc(tensor_courpus_pow, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_neg, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_dot, matmul_strategy()),
    TestFunc(tensor_courpus_dot_general, dog_general_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_transpose, transpose_strategy(), static_argnames={'axes'}),
    TestFunc(tensor_courpus_exp, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_log, broadcasted_elementwise_strategy(1, values=
         positive_float_strategy(min_value=0.1, max_value=10.0)
    )),
    TestFunc(tensor_courpus_sin, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_cos, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_abs, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_sum, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_max, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_min, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_reshape, reshape_strategy(), static_argnames={'shape'}),
    TestFunc(tensor_courpus_broadcast, broadcast_strategy(), static_argnames={'shape'}),
]

dog_elementwise_tests = [
    TestFunc(tensor_courpus_dog_add, dog_post_brodacast_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_sub, dog_post_brodacast_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_mul, dog_post_brodacast_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_div, dog_post_brodacast_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_pow, dog_general_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_neg, dog_general_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_exp, dog_general_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
    TestFunc(tensor_courpus_dog_log, dog_general_strategy(), static_argnames={"lhs_c", "rhs_c", "lhs_b", "rhs_b"}),
]

def get_test_samples(test_set):
    # Strategy to sample functions and arguments
    @st.composite
    def test_sample_strategy(draw):
        # Select a TestFunc
        test_func = draw(st.sampled_from(test_set))

        # Get arguments from the function's strategy
        args = draw(test_func.arg_strategy)

        # Return tuple of (function, args)
        return (test_func.function, args, test_func.static_argnames)

    return test_sample_strategy()

if __name__ == '__main__':
    # Draw a few samples from reshape_strategy()
    for i in range(5):
        arr, shape = reshape_strategy().example()
        print(f"\nSample {i+1}:")
        print(f"Original array shape: {arr.shape}")
        print(f"Reshape target: {shape}")
        print(f"Reshaped array shape: {arr.reshape(shape).shape}")
