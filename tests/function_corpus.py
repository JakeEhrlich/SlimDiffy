from dataclasses import dataclass
from typing import Any, Callable
import slimdiffy.autodiff as ad
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays

@dataclass
class TestFunc:
    function: Callable
    arg_strategy: Any


def shape_strategy():
    # Generate reasonable dimensions for tensor shapes
    sizes = st.integers(min_value=1, max_value=5)
    return st.lists(sizes, min_size=0, max_size=4).map(tuple)

def elementwise_strategy(k: int):
    # Generate shared shape and arrays with same shape
    base = st.shared(shape_strategy())

    @st.composite
    def generate_tensors(draw):
        shape = draw(base)
        tensors = []
        for _ in range(k):
            tensor = draw(arrays(
                np.dtype('float64'),
                shape=shape,
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,  # Restore original value
                    max_value=10.0
                )
            ))
            tensors.append(tensor)
        return tuple(tensors)

    return generate_tensors()

def dog_general_strategy():
    # Generate base shapes and permutations
    batch_shape = st.shared(shape_strategy().map(lambda x: (len(x), x)))
    contract_shape = st.shared(shape_strategy().map(lambda x: (len(x), x)))
    lhs_free_shape = shape_strategy()
    rhs_free_shape = shape_strategy()

    def make_raw_shapes(batch, contract, free):
        n_batch, batch_dims = batch
        n_contract, contract_dims = contract

        # Build dimension lists
        full_shape = list(batch_dims) + list(contract_dims) + list(free)
        full_range = list(range(len(full_shape)))

        # Generate permutations of the indices
        return (st.permutations(full_range).map(lambda p: (
            # Permute the dimensions using the index permutation
            tuple(full_shape[i] for i in p),
            # Map batch indices through permutation
            tuple(p[i] for i in range(n_batch)),
            # Map contract indices through permutation
            tuple(p[i] for i in range(n_batch, n_batch + n_contract)),
            # Map free indices through permutation
            tuple(p[i] for i in range(n_batch + n_contract,
                                    n_batch + n_contract + len(free)))
        )))

    def combine_shapes(bshape, cshape, lhs_free, rhs_free):
        # Get permuted shapes and indices
        lhs_data = make_raw_shapes(bshape, cshape, lhs_free)
        rhs_data = make_raw_shapes(bshape, cshape, rhs_free)

        def combine_data(lhs_perm_data, rhs_perm_data):
            lhs_shape, lhs_b, lhs_c, lhs_f = lhs_perm_data
            rhs_shape, rhs_b, rhs_c, rhs_f = rhs_perm_data

            return (
                np.random.randn(*lhs_shape), # LHS tensor
                np.random.randn(*rhs_shape), # RHS tensor
                tuple(lhs_c),                # LHS contracting dims
                tuple(rhs_c),                # RHS contracting dims
                tuple(lhs_b),                # LHS batch dims
                tuple(rhs_b)                 # RHS batch dims
            )

        return st.tuples(lhs_data, rhs_data).map(
            lambda x: combine_data(*x))

    # Put it all together
    return batch_shape.flatmap(lambda b:
           contract_shape.flatmap(lambda c:
           lhs_free_shape.flatmap(lambda lf:
           rhs_free_shape.flatmap(lambda rf:
           combine_shapes(b, c, lf, rf)))))

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

def reshape_strategy():
    # Generate base shape with non-zero total size
    def valid_shape():
        return st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=1,
            max_size=4
        ).map(tuple)

    base_shape = st.shared(valid_shape())

    def generate_targets(shape):
        total_size = np.prod(shape)

        # Generate valid target shapes that preserve total size
        def valid_targets():
            return st.lists(
                st.integers(min_value=1, max_value=total_size),
                min_size=1,
                max_size=4
            ).map(tuple).filter(lambda x: np.prod(x) == total_size)

        # Add possible -1 dimension
        def add_minus_one(shape):
            if len(shape) <= 1:
                return shape
            pos = np.random.randint(len(shape))
            new_shape = list(shape)
            new_shape[pos] = -1
            return tuple(new_shape)

        target_shape = valid_targets() | valid_targets().map(add_minus_one)

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
            target_shape
        )

    return base_shape.flatmap(generate_targets)

def broadcast_strategy():
    # Generate target shape
    target_shape = shape_strategy()

    def generate_source(target):
        # If target is empty tuple, source must be empty too
        if len(target) == 0:
            return st.tuples(
                arrays(
                    np.dtype('float64'),
                    shape=(),
                    elements=st.floats(
                        allow_infinity=False,
                        allow_nan=False,
                        min_value=-10.0,
                        max_value=10.0
                    )
                ),
                st.just(target)
            )

        # For each dimension, we can either:
        # 1. Keep the original size
        # 2. Replace with size 1 for broadcasting
        # 3. Drop the dimension entirely (but only for leading dims)
        source_dims = []
        for i, size in enumerate(target):
            choice = st.one_of(
                st.just(size),
                st.just(1)
            )
            source_dims.append(choice)

        # Generate length to slice dimensions to
        length = st.integers(min_value=1, max_value=len(target))

        def make_shape(dims, length):
            return tuple(dims[length:])

        return st.tuples(
            length.map(
                lambda x: make_shape(source_dims, x)
            ).flatmap(
                lambda shape: arrays(
                    np.dtype('float64'),
                    shape=shape,
                    elements=st.floats(
                        allow_infinity=False,
                        allow_nan=False,
                        min_value=-10.0,
                        max_value=10.0
                    )
                )
            ),
            st.just(target)
        )

    return target_shape.flatmap(generate_source)

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
        return x + (epsilon if x >= 0 else -epsilon) if abs(x) < epsilon else x
    return strategy()

def positive_float_strategy(min_value=0.1, max_value=10.0, epsilon=1e-3):
    @st.composite
    def strategy(draw):
        x = draw(st.floats(
            allow_infinity=False,
            allow_nan=False,
            min_value=min_value,
            max_value=max_value
        ))
        return max(x, epsilon)  # Ensure value is at least epsilon
    return strategy()

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
            return draw(elementwise_strategy(k))

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

def dog_post_brodacast_strategy():
    # Get dot general inputs
    base_dog = st.shared(dog_general_strategy())

    def generate_broadcast(base):
        x_orig, y_orig, c1, c2, b1, b2 = base
        x = x_orig if isinstance(x_orig, np.ndarray) else np.array(x_orig)
        y = y_orig if isinstance(y_orig, np.ndarray) else np.array(y_orig)

        # Calculate output shape of dot_general
        out_shape = tuple(x.shape[i] for i in b1) + \
                   tuple(d for i, d in enumerate(x.shape)
                        if i not in c1 and i not in b1) + \
                   tuple(d for i, d in enumerate(y.shape)
                        if i not in c2 and i not in b2)

        final_shape = generate_broadcast_shape_strategy(out_shape)

        return st.tuples(
            st.just(x),
            st.just(y),
            st.just(c1),
            st.just(c2),
            st.just(b1),
            st.just(b2),
            arrays(
                np.dtype('float64'),
                shape=final_shape,
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            )
        )

    return base_dog.flatmap(generate_broadcast)


def dog_pre_broadcast_left_strategy():
    # Get dot general inputs first
    base_dog = st.shared(dog_general_strategy())

    def generate_broadcast(base):
        x_orig, y_orig, c1, c2, b1, b2 = base
        x = x_orig if isinstance(x_orig, np.ndarray) else np.array(x_orig)

        # Generate broadcast shape for x
        final_shape = st.shared(shape_strategy())
        def check_broadcast(shape):
            try:
                np.broadcast_shapes(x.shape, shape)
                return True
            except ValueError:
                return False

        return st.tuples(
            arrays(
                np.dtype('float64'),
                shape=final_shape,
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            ).filter(lambda z: check_broadcast(z.shape)),
            st.just(y_orig),
            st.just(c1),
            st.just(c2),
            st.just(b1),
            st.just(b2)
        )

    return base_dog.flatmap(generate_broadcast)

def dog_pre_broadcast_right_strategy():
    # Get dot general inputs first
    base_dog = st.shared(dog_general_strategy())

    def generate_broadcast(base):
        x_orig, y_orig, c1, c2, b1, b2 = base
        y = y_orig if isinstance(y_orig, np.ndarray) else np.array(y_orig)

        # Generate broadcast shape for y
        final_shape = st.shared(shape_strategy())
        def check_broadcast(shape):
            try:
                np.broadcast_shapes(y.shape, shape)
                return True
            except ValueError:
                return False

        return st.tuples(
            st.just(x_orig),
            arrays(
                np.dtype('float64'),
                shape=final_shape,
                elements=st.floats(
                    allow_infinity=False,
                    allow_nan=False,
                    min_value=-10.0,
                    max_value=10.0
                )
            ).filter(lambda z: check_broadcast(z.shape)),
            st.just(c1),
            st.just(c2),
            st.just(b1),
            st.just(b2)
        )

    return base_dog.flatmap(generate_broadcast)

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
    return ad.sum(x).dot_general(z, lhs_contracting_dims=lhs_c,
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
    return x.dot_general(y, lhs_contracting_dims=lhs_c,
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
    return x.reshape(shape)

def tensor_courpus_broadcast(x, shape):
    return x.broadcast_to(shape)

def tensor_courpus_add_mul(x, y):
    return (x + y) * (x + y)

def tensor_courpus_add_div(x, y):
    return (x + y) / (x + y)

def tensor_courpus_add_pow(x, y):
    return (x + y) ** 2

def tensor_courpus_add_exp(x, y):
    return ad.exp(x + y)

def tensor_courpus_add_log(x, y):
    return ad.log(x + y)

def tensor_courpus_add_sin(x, y):
    return ad.sin(x + y)

def tensor_courpus_add_cos(x, y):
    return ad.cos(x + y)

def tensor_courpus_sub_mul(x, y):
    return (x - y) * (x - y)

def tensor_courpus_sub_div(x, y):
    return (x - y) / (x - y)

def tensor_courpus_sub_pow(x, y):
    return (x - y) ** 2

def tensor_courpus_sub_exp(x, y):
    return ad.exp(x - y)

def tensor_courpus_sub_log(x, y):
    return ad.log(x - y)

def tensor_courpus_sub_sin(x, y):
    return ad.sin(x - y)

def tensor_courpus_sub_cos(x, y):
    return ad.cos(x - y)

def tensor_courpus_mul_div(x, y):
    return (x * y) / (x * y)

def tensor_courpus_mul_pow(x, y):
    return (x * y) ** 2

def tensor_courpus_mul_exp(x, y):
    return ad.exp(x * y)

def tensor_courpus_mul_log(x, y):
    return ad.log(x * y)

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
    TestFunc(tensor_courpus_add_div, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_log, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_add_cos, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_add_min, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_add_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_mul, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_div, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_log, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sub_cos, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_sub_min, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_sub_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_div, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_pow, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_exp, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_log, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_sin, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_mul_cos, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_mul_min, broadcasted_elementwise_strategy(2)),
    #TestFunc(tensor_courpus_mul_max, broadcasted_elementwise_strategy(2)),
    TestFunc(tensor_courpus_sin_cos, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_cos_sin, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_exp_log, broadcasted_elementwise_strategy(1)),
    TestFunc(tensor_courpus_log_exp, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_min_max, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_max_min, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_min_exp, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_min_log, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_min_sin, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_min_cos, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_max_exp, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_max_log, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_max_sin, broadcasted_elementwise_strategy(1)),
    # TestFunc(tensor_courpus_max_cos, broadcasted_elementwise_strategy(1)),
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
    #TestFunc(tensor_courpus_dot_general, dog_general_strategy()),
    #TestFunc(tensor_courpus_transpose, transpose_strategy()),
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
    #TestFunc(tensor_courpus_reshape, reshape_strategy()),
    #TestFunc(tensor_courpus_broadcast, broadcast_strategy()),
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
        return (test_func.function, args)

    return test_sample_strategy()
