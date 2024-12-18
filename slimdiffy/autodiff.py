# Copyright (c) 2024 Jake Ehrlich
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # type: ignore
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union, Callable
import inspect
from enum import Enum
import slimdiffy.pytree as pt

@dataclass
class Var:
    arg_index: int
    dtype: np.dtype
    shape: Tuple[int, ...]

@dataclass
class Literal:
    value: np.ndarray

    @property
    def dtype(self) -> np.dtype:
        return self.value.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

@dataclass
class Lambda:
    args: Tuple[pt.Node, ...] # A list of PyTrees whos leaves are argument indexes
    equations: List['Expr']
    result: pt.Node # a PyTree whos leaves are equation indexes

    def render(self, indent: int = 0) -> str:
        # Start with lambda args
        result = " " * indent + "{" + ", ".join(f"a{i}" for i, _ in enumerate(self.args)) + " |\n"
        indent += 2
        # Add equations
        for i, eq in enumerate(self.equations):
            if isinstance(eq, Op):
                eq_str = f"v{i} = {eq.op.value}"
                # Get operator inputs
                input_strs = [f"v{j}" for j in eq.inputs]
                # Add axes for reduction ops
                if eq.op in (OpType.SUM, OpType.PROD, OpType.MIN, OpType.MAX):
                    eq_str += f"({', '.join(input_strs)}, axes={eq.metadata['axes']}"
                    if eq.metadata.get('keepdims', False):
                        eq_str += f", keepdims={eq.metadata['keepdims']}"
                    eq_str += ")"
                else:
                    eq_str += f"({', '.join(input_strs)})"
            elif isinstance(eq, Literal):
                eq_str = f"v{i} = {eq.value}"
            elif isinstance(eq, Var):
                path = []
                found = False
                target_idx: int = eq.arg_index
                tree_idx = None
                cur_tree_idx = 0
                def make_path_string(key_path: Tuple[Union[str, int], ...], leaf_value: Any) -> None:
                    nonlocal found, path, tree_idx, cur_tree_idx
                    if not found and leaf_value == target_idx:
                        for key in key_path:
                            if isinstance(key, int):
                                path.append(f"[{key}]")
                            else:
                                assert isinstance(key, str)
                                if key.isidentifier():
                                    path.append(f".{key}")
                                else:
                                    path.append(f"[{repr(key)}]")
                        tree_idx = cur_tree_idx
                        found = True

                for i, tree in enumerate(self.args):
                    cur_tree_idx = i
                    pt.mapkeys(make_path_string, tree)
                assert found and tree_idx is not None
                path_str = ''.join(path)
                eq_str = f"v{i} = a{tree_idx}{path_str}"
            else:
                raise ValueError("")
            result += " " * indent + eq_str
            result += '\n'

        # Add result using PyTreeNode render
        result += " " * indent + "return " + self.result.render(indent, lambda x: f"v{x}")
        result += " " * indent + "\n}"
        return result

    def __str__(self) -> str:
        return self.render()


class OpType(Enum):
    # Basic arithmetic
    ADD = 'add'
    SUB = 'sub'
    MUL = 'mul'
    DIV = 'div'
    POW = 'pow'
    NEG = 'neg'
    BROADCAST = 'broadcast'

    # Comparisons
    LT = 'lt'
    GT = 'gt'
    MAXIMUM = 'maximum'
    MINIMUM = 'minimum'

    # Linear algebra
    DOT = 'dot'
    TRANSPOSE = 'transpose'
    RESHAPE = 'reshape'

    # Reductions
    SUM = 'sum'
    PROD = 'prod'
    MIN = 'min'
    MAX = 'max'

    # Elementwise functions
    EXP = 'exp'
    LOG = 'log'
    SIN = 'sin'
    COS = 'cos'
    ABS = 'abs'

    # IO operations
    IO_CALLBACK = 'io_callback'

@dataclass
class Op:
    op: OpType
    inputs: List[int]  # List of indices into equations list
    dtype: np.dtype
    shape: Tuple[int, ...]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

Expr = Union[Op, Literal, Var]



class TracerSupervisor:
    def __init__(self):
        self.equations: List[Expr] = []

    def add_equation(self, expr: Expr) -> int:
        self.equations.append(expr)
        return len(self.equations) - 1

    def create_lambda(self, args: Tuple[pt.Node, ...], result: pt.Node) -> Lambda:
        assert all(isinstance(arg, pt.Node) for arg in args)
        assert isinstance(result, pt.Node)
        return Lambda(args, self.equations, result)

class Tracer:
    def __init__(self, expr: Expr, supervisor: TracerSupervisor):
        self.supervisor = supervisor
        self.idx = supervisor.add_equation(expr)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.supervisor.equations[self.idx].shape

    @property
    def dtype(self) -> np.dtype:
        return self.supervisor.equations[self.idx].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def _binary_op(self, other, op: OpType) -> 'Tracer':
        other = _ensure_tracer(other, self.supervisor)
        self_expr = self.supervisor.equations[self.idx]
        other_expr = self.supervisor.equations[other.idx]

        try:
            out_shape = np.broadcast_shapes(self_expr.shape, other_expr.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {self_expr.shape} and {other_expr.shape}")

        # Insert explicit broadcasts if needed
        a = self
        b = other
        if self_expr.shape != out_shape:
            a = self.broadcast_to(out_shape)
        if other_expr.shape != out_shape:
            b = other.broadcast_to(out_shape)

        return Tracer(Op(op, [a.idx, b.idx], self_expr.dtype, out_shape), self.supervisor)

    def _unary_op(self, op: OpType) -> 'Tracer':
        self_expr = self.supervisor.equations[self.idx]
        return Tracer(Op(op, [self.idx], self_expr.dtype, self_expr.shape), self.supervisor)

    def broadcast_to(self, shape: Tuple[int, ...]) -> 'Tracer':
        self_expr = self.supervisor.equations[self.idx]
        # If shapes match exactly, no need to broadcast
        if self_expr.shape == shape:
            return self
        try:
            np.broadcast_shapes(self_expr.shape, shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shape {self_expr.shape} to {shape}")
        return Tracer(Op(OpType.BROADCAST, [self.idx], self_expr.dtype, shape), self.supervisor)

    def reshape(self, *shape: int) -> 'Tracer':
        """Reshape array to new shape"""
        self_expr = self.supervisor.equations[self.idx]
        if -1 in shape:
            # Calculate size of -1 dimension
            known_size = 1
            unknown_idx = None
            for i, s in enumerate(shape):
                if s == -1:
                    if unknown_idx is not None:
                        raise ValueError("Only one -1 allowed in reshape")
                    unknown_idx = i
                else:
                    known_size *= s
            assert isinstance(unknown_idx, int)
            full_size = np.prod(self_expr.shape)
            if full_size % known_size != 0:
                raise ValueError(f"Cannot reshape array of size {full_size} into shape {shape}")
            shape_list = list(shape)
            shape_list[unknown_idx] = int(full_size) // known_size
            shape = tuple(shape_list)
        else:
            # Verify shapes are compatible
            if np.prod(shape) != np.prod(self_expr.shape):
                raise ValueError(f"Cannot reshape array of size {np.prod(self_expr.shape)} into shape {shape}")
            # If shape matches exactly, no need to reshape
            if shape == self_expr.shape:
                return self
        return Tracer(Op(OpType.RESHAPE, [self.idx], self_expr.dtype, shape), self.supervisor)

    def __add__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        other_expr = self.supervisor.equations[other.idx]
        self_expr = self.supervisor.equations[self.idx]

        if isinstance(other_expr, Literal) and np.all(other_expr.value == 0.0):
            return self
        if isinstance(self_expr, Literal) and np.all(self_expr.value == 0.0):
            return other

        return self._binary_op(other, OpType.ADD)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        other_expr = self.supervisor.equations[other.idx]
        self_expr = self.supervisor.equations[self.idx]

        if isinstance(other_expr, Literal):
            if np.all(other_expr.value == 1.0):
                return self
            if np.all(other_expr.value == 0.0):
                return other
        if isinstance(self_expr, Literal):
            if np.all(self_expr.value == 1.0):
                return other
            if np.all(self_expr.value == 0.0):
                return self

        return self._binary_op(other, OpType.MUL)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary_op(other, OpType.DIV)

    def __rtruediv__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        return other._binary_op(self, OpType.DIV)

    def __matmul__(self, other):
        # Convert matmul to dot_general with appropriate contracting dimensions
        return self.dot_general(other,
                              lhs_contracting_dims=(1,),
                              rhs_contracting_dims=(0,),
                              lhs_batch_dims=(),
                              rhs_batch_dims=())

    def __rmatmul__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        return other.__matmul__(self)

    def dot_general(self, other,
                   lhs_contracting_dims: Tuple[int, ...],
                   rhs_contracting_dims: Tuple[int, ...],
                   lhs_batch_dims: Tuple[int, ...],
                   rhs_batch_dims: Tuple[int, ...]) -> 'Tracer':
        other = _ensure_tracer(other, self.supervisor)
        self_expr = self.supervisor.equations[self.idx]
        other_expr = self.supervisor.equations[other.idx]

        # Check contracting dimensions match in length
        if len(lhs_contracting_dims) != len(rhs_contracting_dims):
            raise ValueError("Number of lhs and rhs contracting dims must match")

        # Check batch dimensions match in length
        if len(lhs_batch_dims) != len(rhs_batch_dims):
            raise ValueError("Number of lhs and rhs batch dims must match")

        # Check contracting dimension sizes match
        for l, r in zip(lhs_contracting_dims, rhs_contracting_dims):
            if self_expr.shape[l] != other_expr.shape[r]:
                breakpoint()
                raise ValueError(f"Contracting dimension mismatch: {self_expr.shape[l]} != {other_expr.shape[r]}")

        # Check batch dimension sizes match
        for l, r in zip(lhs_batch_dims, rhs_batch_dims):
            if self_expr.shape[l] != other_expr.shape[r]:
                raise ValueError(f"Batch dimension mismatch: {self_expr.shape[l]} != {other_expr.shape[r]}")

        # Calculate output shape:
        # 1. Start with batch dimensions from lhs
        out_shape = tuple(self_expr.shape[i] for i in lhs_batch_dims)
        # 2. Add remaining non-contracting dims from lhs
        out_shape += tuple(d for i, d in enumerate(self_expr.shape)
                         if i not in lhs_contracting_dims and i not in lhs_batch_dims)
        # 3. Add remaining non-contracting dims from rhs
        out_shape += tuple(d for i, d in enumerate(other_expr.shape)
                         if i not in rhs_contracting_dims and i not in rhs_batch_dims)

        metadata = {
            'lhs_contracting_dims': lhs_contracting_dims,
            'rhs_contracting_dims': rhs_contracting_dims,
            'lhs_batch_dims': lhs_batch_dims,
            'rhs_batch_dims': rhs_batch_dims
        }

        return Tracer(Op(OpType.DOT, [self.idx, other.idx], self_expr.dtype, out_shape, metadata),
                     self.supervisor)

    def __sub__(self, other):
        return self._binary_op(other, OpType.SUB)

    def __rsub__(self, other):
        return self._binary_op(other, OpType.SUB)

    def __pow__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        other_expr = self.supervisor.equations[other.idx]
        # Check for special case x**1 = x
        if isinstance(other_expr, Literal) and np.all(other_expr.value == 1.0):
            return self
        return self._binary_op(other, OpType.POW)

    def __rpow__(self, other):
        return self._binary_op(other, OpType.POW)

    def __neg__(self):
        return self._unary_op(OpType.NEG)

    def __lt__(self, other):
        return self._binary_op(other, OpType.LT)

    def __gt__(self, other):
        return self._binary_op(other, OpType.GT)

    def maximum(self, other):
        return self._binary_op(other, OpType.MAXIMUM)

    def minimum(self, other):
        return self._binary_op(other, OpType.MINIMUM)

    def transpose(self, *axes: int) -> 'Tracer':
        """Permute dimensions according to axes"""
        self_expr = self.supervisor.equations[self.idx]
        if len(axes) == 0:
            # Default is to reverse dimensions
            axes = tuple(range(len(self_expr.shape)-1, -1, -1))
        elif len(axes) != len(self_expr.shape):
            raise ValueError("axes don't match array dimensions")

        # Check for identity permutation
        if axes == tuple(range(len(self_expr.shape))):
            return self

        # Check for duplicate axes
        if len(set(axes)) != len(axes):
            raise ValueError("axes contains duplicate values")

        # Check all axes are valid
        n_dims = len(self_expr.shape)
        valid_axes = set(range(n_dims))
        if not set(axes).issubset(valid_axes):
            raise ValueError(f"axes must be integers in range [0, {n_dims-1}]")

        # Calculate new shape after permutation
        new_shape = tuple(self_expr.shape[i] for i in axes)
        metadata = {'axes': axes}
        return Tracer(Op(OpType.TRANSPOSE, [self.idx], self_expr.dtype, new_shape, metadata), self.supervisor)

    @property
    def T(self):
        self_expr = self.supervisor.equations[self.idx]
        if len(self_expr.shape) != 2:
            raise ValueError("T property requires 2D array")
        return self.transpose(1, 0)

    def sum(self, axis=None, keepdims=False):
        return self._reduction_op(OpType.SUM, axis, keepdims)

    def prod(self, axis=None, keepdims=False):
        return self._reduction_op(OpType.PROD, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return self._reduction_op(OpType.MAX, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return self._reduction_op(OpType.MIN, axis, keepdims)

    def _reduction_op(self, op: OpType, axis=None, keepdims=False) -> 'Tracer':
        self_expr = self.supervisor.equations[self.idx]

        # Special case - reduction over empty axes is identity
        if axis == ():
            return self

        if axis is None:
            axes = tuple(range(len(self_expr.shape)))
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis
        if keepdims:
            new_shape = tuple(1 if i in axes else s for i, s in enumerate(self_expr.shape))
        else:
            new_shape = tuple(s for i, s in enumerate(self_expr.shape) if i not in axes)
        metadata = {'axes': axes, 'keepdims': keepdims}
        return Tracer(Op(op, [self.idx], self_expr.dtype, new_shape, metadata), self.supervisor)

def io_callback(x, fn):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.IO_CALLBACK, [x.idx], self_expr.dtype, self_expr.shape,
                        metadata={'callback': fn}), x.supervisor)
    return fn(x)

def reshape(x, shape):
    if isinstance(x, Tracer):
        return x.reshape(*shape)
    return np.reshape(x, shape)

def transpose(x, axes):
    if isinstance(x, Tracer):
        return x.transpose(*axes)
    return np.transpose(x, axes)

def dot_general(x, y, *, lhs_contracting_dims, rhs_contracting_dims, lhs_batch_dims, rhs_batch_dims):
    if isinstance(x, Tracer):
        return x.dot_general(y, lhs_contracting_dims=lhs_contracting_dims,
                        rhs_contracting_dims=rhs_contracting_dims,
                        lhs_batch_dims=lhs_batch_dims,
                        rhs_batch_dims=rhs_batch_dims)
    elif isinstance(y, Tracer):
        x = _ensure_tracer(x, y.supervisor)
        return x.dot_general(y,
                        lhs_contracting_dims=lhs_contracting_dims,
                        rhs_contracting_dims=rhs_contracting_dims,
                        lhs_batch_dims=lhs_batch_dims,
                        rhs_batch_dims=rhs_batch_dims)
    else:
        return batch_contract_einsum(x, y, lhs_contracting_dims, rhs_contracting_dims,
                                   lhs_batch_dims=lhs_batch_dims,
                                   rhs_batch_dims=rhs_batch_dims)

def broadcast_to(x, shape):
    if isinstance(x, Tracer):
        return x.broadcast_to(shape)
    return np.broadcast_to(x, shape)

def minimum(x, other):
    if isinstance(x, Tracer):
        return x.minimum(other)
    if isinstance(other, Tracer):
        return other.minimum(x)
    return np.minimum(x, other)

def maximum(x, other):
    if isinstance(x, Tracer):
        return x.maximum(other)
    if isinstance(other, Tracer):
        return other.maximum(x)
    return np.maximum(x, other)

def sum(x, axis=None, keepdims=False):
    if isinstance(x, Tracer):
        return x.sum(axis, keepdims=keepdims)
    return np.sum(x, axis=axis, keepdims=keepdims)

def prod(x, axis=None, keepdims=False):
    if isinstance(x, Tracer):
        return x.prod(axis, keepdims=keepdims)
    return np.prod(x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    if isinstance(x, Tracer):
        return x.min(axis, keepdims=keepdims)
    return np.min(x, axis=axis, keepdims=keepdims)

def max(x, axis=None, keepdims=False):
    if isinstance(x, Tracer):
        return x.max(axis, keepdims=keepdims)
    return np.max(x, axis=axis, keepdims=keepdims)

def exp(x):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.EXP, [x.idx], self_expr.dtype, self_expr.shape), x.supervisor)
    return np.exp(x)

def log(x):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.LOG, [x.idx], self_expr.dtype, self_expr.shape), x.supervisor)
    return np.log(x)

def sin(x):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.SIN, [x.idx], self_expr.dtype, self_expr.shape), x.supervisor)
    return np.sin(x)

def cos(x):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.COS, [x.idx], self_expr.dtype, self_expr.shape), x.supervisor)
    return np.cos(x)

def abs(x):
    if isinstance(x, Tracer):
        self_expr = x.supervisor.equations[x.idx]
        return Tracer(Op(OpType.ABS, [x.idx], self_expr.dtype, self_expr.shape), x.supervisor)
    return np.abs(x)

def _ensure_tracer(x, supervisor):
    if isinstance(x, Tracer):
        return x
    if isinstance(x, (int, float)):
        x = np.array(x)
    assert type(x) is np.ndarray
    return Tracer(Literal(x), supervisor)

def batch_contract_einsum(a, b, lhs_dims, rhs_dims, lhs_batch_dims, rhs_batch_dims):
    """
    Perform a batched tensor contraction between tensors a and b using einsum.

    Like batch_contract but using einsum notation instead of explicit reshape/transpose.
    """
    # Get non-contract, non-batch dims
    lhs_free = [i for i in range(len(a.shape)) if i not in lhs_dims and i not in lhs_batch_dims]
    rhs_free = [i for i in range(len(b.shape)) if i not in rhs_dims and i not in rhs_batch_dims]

    # Build dimension labels using ASCII lowercase letters
    dim_labels = iter('ijklmnopqrstuvwxyzabcdefgh')

    # Assign labels to each dimension type
    batch_labels = [next(dim_labels) for _ in lhs_batch_dims]
    lhs_free_labels = [next(dim_labels) for _ in lhs_free]
    rhs_free_labels = [next(dim_labels) for _ in rhs_free]
    contract_labels = [next(dim_labels) for _ in lhs_dims]

    # Build lhs subscript by assigning labels to the right dims
    lhs_labels = ["!"] * len(a.shape)
    for i, label in zip(lhs_batch_dims, batch_labels):
        lhs_labels[i] = label
    for i, label in zip(lhs_free, lhs_free_labels):
        lhs_labels[i] = label
    for i, label in zip(lhs_dims, contract_labels):
        lhs_labels[i] = label

    # Build rhs subscript similarly
    rhs_labels = ["!"] * len(b.shape)
    for i, label in zip(rhs_batch_dims, batch_labels):
        rhs_labels[i] = label
    for i, label in zip(rhs_free, rhs_free_labels):
        rhs_labels[i] = label
    for i, label in zip(rhs_dims, contract_labels):
        rhs_labels[i] = label

    # Build output subscript from batch + free dims
    out_labels = batch_labels + lhs_free_labels + rhs_free_labels

    # Join subscripts with comma
    einsum_str = ''.join(lhs_labels) + ',' + ''.join(rhs_labels) + '->' + ''.join(out_labels)

    return np.einsum(einsum_str, a, b)

def batch_contract(a, b, lhs_dims, rhs_dims, lhs_batch_dims, rhs_batch_dims):
    """
    Perform a batched tensor contraction between tensors a and b.

    Parameters:
    -----------
    a : np.ndarray
        Left tensor
    b : np.ndarray
        Right tensor
    lhs_dims : list[int]
        Indices of dimensions in a to contract
    rhs_dims : list[int]
        Matching indices of dimensions in b to contract
    lhs_batch_dims : tuple[int]
        Indices of batch dimensions in left tensor
    rhs_batch_dims : tuple[int]
        Indices of batch dimensions in right tensor

    Returns:
    --------
    np.ndarray
        Contracted tensor

    Example:
    --------
    # Contract along last dimension:
    # a: (batch=3, free=5, contract=7)
    # b: (batch=3, free=2, contract=7)
    # result: (batch=3, free_a=5, free_b=2)
    # batch_contract(a, b, lhs_dims=[2], rhs_dims=[2], lhs_batch_dims=[0], rhs_batch_dims=[0])
    """
    # Verify contracting dimensions match in size
    for l, r in zip(lhs_dims, rhs_dims):
        if a.shape[l] != b.shape[r]:
            raise ValueError(f"Contracting dimensions must match: "
                           f"a dim {l} has size {a.shape[l]}, "
                           f"b dim {r} has size {b.shape[r]}")

    # Verify batch dimensions match in length and size
    if len(lhs_batch_dims) != len(rhs_batch_dims):
        raise ValueError("Number of batch dimensions must match")

    for l, r in zip(lhs_batch_dims, rhs_batch_dims):
        if a.shape[l] != b.shape[r]:
            raise ValueError(f"Batch dimensions must match: "
                           f"a dim {l} has size {a.shape[l]}, "
                           f"b dim {r} has size {b.shape[r]}")

    # Identify dimensions that aren't contracted or batched
    a_free = tuple(i for i in range(a.ndim)
              if i not in lhs_dims and
              i not in lhs_batch_dims)
    b_free = tuple(i for i in range(b.ndim)
              if i not in rhs_dims and
              i not in rhs_batch_dims)

    # Create permutation: [batch_dims, free_dims, contract_dims]
    a_perm = (tuple(lhs_batch_dims) +
              a_free +
              lhs_dims)
    b_perm = (tuple(rhs_batch_dims) +
              b_free +
              rhs_dims)

    a_transposed = np.transpose(a, a_perm)
    b_transposed = np.transpose(b, b_perm)

    # Get shapes for each part
    n_batch = len(lhs_batch_dims)
    n_contract = len(lhs_dims)

    a_shape = a_transposed.shape
    b_shape = b_transposed.shape

    # Reshape to combine batch dims and free dims
    a_reshaped = a_transposed.reshape(
        (-1,) +
        (np.prod(a_shape[n_batch:-n_contract], dtype=np.int64),) +
        (np.prod(a_shape[-n_contract:], dtype=np.int64),))

    b_reshaped = b_transposed.reshape(
        (-1,) +
        (np.prod(b_shape[n_batch:-n_contract], dtype=np.int64),) +
        (np.prod(b_shape[-n_contract:], dtype=np.int64),))

    # Transpose b to get contract dims first for matmul
    b_reshaped = np.transpose(b_reshaped, (0, 2, 1))

    # Perform batched matrix multiplication
    result = np.matmul(a_reshaped, b_reshaped)

    # Reshape back to full dimension tensor
    final_shape = (a_shape[:n_batch] +
                  a_shape[n_batch:-n_contract] +
                  b_shape[n_batch:-n_contract])
    return result.reshape(final_shape)

class Interpreter:
    def __init__(self, inputs: Tuple[pt.Node, ...]):
        self.inputs = inputs
        self.flat_inputs = []
        self.results = {}

    def __call__(self, lambda_expr: Lambda) -> Any:
        """Basic interpreter for equations using visitor pattern"""
        # Extract flat array inputs from PyTrees
        self.flat_inputs = jit.get_values_from_index_tree(self.inputs, lambda_expr.args)

        # Process equations
        for i, eq in enumerate(lambda_expr.equations):
            self.results[i] = self.visit(eq)
            print(f"Index {i}: {self.results[i]}")  # Debug print

        # Use pytree to construct result
        def get_result(idx: Union[int, tuple, Tracer]) -> Any:
            print(f"Getting result for idx: {idx}")  # Debug print
            print(f"Available indices: {list(self.results.keys())}")  # Debug print
            if isinstance(idx, tuple):
                # Handle tuple indices by recursively getting each element
                return tuple(get_result(i) for i in idx)
            if isinstance(idx, Tracer):
                # If we get a Tracer, return it directly
                return idx
            if idx not in self.results:
                raise KeyError(f"Index {idx} not found in results. Available indices: {list(self.results.keys())}")
            return self.results[idx]

        result_tree = pt.map(get_result, lambda_expr.result)
        return result_tree.to_value()

    def visit(self, expr: Expr) -> Any:
        method = f'visit_{type(expr).__name__.lower()}'
        visitor = getattr(self, method)
        return visitor(expr)

    def visit_literal(self, expr: Literal) -> Any:
        return expr.value

    def visit_var(self, expr: Var) -> Any:
        return self.flat_inputs[expr.arg_index]

    def visit_op(self, expr: Op) -> Any:
        method = f'visit_op_{expr.op.name.lower()}'
        visitor = getattr(self, method)
        return visitor(expr)

    def visit_op_add(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] + self.results[expr.inputs[1]]

    def visit_op_mul(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] * self.results[expr.inputs[1]]

    def visit_op_sub(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] - self.results[expr.inputs[1]]

    def visit_op_div(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] / self.results[expr.inputs[1]]

    def visit_op_pow(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] ** self.results[expr.inputs[1]]

    def visit_op_neg(self, expr: Op) -> Any:
        return -self.results[expr.inputs[0]]

    def visit_op_lt(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] < self.results[expr.inputs[1]]

    def visit_op_gt(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] > self.results[expr.inputs[1]]

    def visit_op_maximum(self, expr: Op) -> Any:
        return np.maximum(self.results[expr.inputs[0]], self.results[expr.inputs[1]])

    def visit_op_minimum(self, expr: Op) -> Any:
        return np.minimum(self.results[expr.inputs[0]], self.results[expr.inputs[1]])

    def visit_op_exp(self, expr: Op) -> Any:
        return np.exp(self.results[expr.inputs[0]])

    def visit_op_log(self, expr: Op) -> Any:
        return np.log(self.results[expr.inputs[0]])

    def visit_op_sin(self, expr: Op) -> Any:
        return np.sin(self.results[expr.inputs[0]])

    def visit_op_cos(self, expr: Op) -> Any:
        return np.cos(self.results[expr.inputs[0]])

    def visit_op_abs(self, expr: Op) -> Any:
        return np.abs(self.results[expr.inputs[0]])

    def visit_op_dot(self, expr: Op) -> Any:
        a = self.results[expr.inputs[0]]
        b = self.results[expr.inputs[1]]

        # Extract dimensions from metadata
        lhs_c = expr.metadata['lhs_contracting_dims']
        rhs_c = expr.metadata['rhs_contracting_dims']
        lhs_b = expr.metadata['lhs_batch_dims']
        rhs_b = expr.metadata['rhs_batch_dims']

        # Full dot general using batch_contract
        return batch_contract_einsum(a, b, lhs_c, rhs_c, lhs_batch_dims=lhs_b, rhs_batch_dims=rhs_b)

    def visit_op_transpose(self, expr: Op) -> Any:
        axes = expr.metadata.get('axes')
        if axes:
            return np.transpose(self.results[expr.inputs[0]], axes)
        return np.transpose(self.results[expr.inputs[0]])

    def visit_op_reshape(self, expr: Op) -> Any:
        return np.reshape(self.results[expr.inputs[0]], expr.shape)

    def visit_op_broadcast(self, expr: Op) -> Any:
        x = np.asarray(self.results[expr.inputs[0]])
        return np.broadcast_to(x, expr.shape)

    def visit_op_io_callback(self, expr: Op) -> Any:
        input_values = [self.results[i] for i in expr.inputs]
        expr.metadata['callback'](*input_values)
        return np.nan

    def visit_op_sum(self, expr: Op) -> Any:
        return np.sum(self.results[expr.inputs[0]],
                     axis=expr.metadata['axes'],
                     keepdims=expr.metadata.get('keepdims', False))

    def visit_op_prod(self, expr: Op) -> Any:
        return np.prod(self.results[expr.inputs[0]],
                      axis=expr.metadata['axes'],
                      keepdims=expr.metadata.get('keepdims', False))

    def visit_op_min(self, expr: Op) -> Any:
        return np.min(self.results[expr.inputs[0]],
                     axis=expr.metadata['axes'],
                     keepdims=expr.metadata.get('keepdims', False))

    def visit_op_max(self, expr: Op) -> Any:
        return np.max(self.results[expr.inputs[0]],
                     axis=expr.metadata['axes'],
                     keepdims=expr.metadata.get('keepdims', False))

@dataclass
class ArgSpec:
    dtype: Any
    shape: Tuple[int, ...]

    @staticmethod
    def from_pytree(pytree: pt.Node) -> pt.Node:
        """Transforms a PyTreeNode of np.ndarrays into a PyTreeNode of ArgSpecs"""
        def to_argspec(leaf_value) -> ArgSpec:
            if type(leaf_value) is not np.ndarray:
                leaf_value = np.array(leaf_value)
            return ArgSpec(leaf_value.dtype, leaf_value.shape)
        return pt.map(to_argspec, pytree)

class Transform:
    """Base class for function transformations like jit and grad"""
    def __init__(self, fn, *, transforms=(), static_argnames=None):
        # If fn is already a Transform, compose the transforms
        if isinstance(fn, Transform):
            self.fn = fn.fn
            self.signature = fn.signature
            self.transforms = fn.transforms + transforms
            self.static_argnames = fn.static_argnames | (static_argnames or set())
        else:
            self.fn = fn
            self.signature = inspect.signature(fn)
            self.transforms = transforms
            self.static_argnames = static_argnames or set()
        self.expr_cache = {}

    @staticmethod
    def unindex_pytree(index_tree: pt.Node, values: Tuple[Any, ...]) -> pt.Node:
        def unindex(idx: int):
            return values[idx]
        return pt.map(unindex, index_tree)

    @staticmethod
    def index_pytrees(*pytrees: pt.Node) -> Tuple[pt.Node, ...]:
        """Replaces leaves in pytrees with unique indices"""
        next_idx = 0
        def assign_index(trees: Any) -> int:
            nonlocal next_idx
            idx = next_idx
            next_idx += 1
            return idx
        return tuple(pt.map(assign_index, tree) for tree in pytrees)

    @staticmethod
    def get_values_from_index_tree(value_tree: Union[pt.Node, Tuple], index_tree: Union[pt.Node, Tuple]) -> Tuple[Any, ...]:
        """Extracts values from value_tree based on indices in index_tree"""
        values: List[Any] = []
        def assign_value(value: Any, index: int) -> None:
            nonlocal values
            while len(values) <= index:
                values.append(None)
            values[index] = value

        if isinstance(value_tree, tuple) and isinstance(index_tree, tuple):
            for v, i in zip(value_tree, index_tree):
                pt.map(assign_value, v, i)
        else:
            assert isinstance(value_tree, pt.Node)
            assert isinstance(index_tree, pt.Node)
            pt.map(assign_value, value_tree, index_tree)
        return tuple(values)

    def get_expr(self, *arg_specs, pred=lambda _, __: True):
        """Get expression graph for function given input specs"""
        # Create key from arg_specs for caching
        frozen_specs = pt.freeze(arg_specs)
        if frozen_specs in self.expr_cache:
            return self.expr_cache[frozen_specs]

        supervisor = TracerSupervisor()
        full_tree = pt.from_sequence(arg_specs, pred=pred)
        index_tree, = self.index_pytrees(full_tree)
        def make_var(arg_spec, arg_index):
            return Tracer(Var(arg_index, arg_spec.dtype, arg_spec.shape), supervisor)
        trace_vars = pt.map(make_var, full_tree, index_tree).to_value()
        result = self.fn(*trace_vars)
        index_tuple = index_tree.to_sequence(False)
        assert isinstance(index_tuple, Tuple)
        if isinstance(result, Tracer):
            lambda_expr = supervisor.create_lambda(index_tuple, pt.leaf(result.idx))
        else:
            pytree = pt.from_value(result)
            pytree = pt.map(lambda x: x.idx, pytree)
            assert isinstance(pytree, pt.Node)
            lambda_expr = supervisor.create_lambda(index_tuple, pytree)

        # Cache and return the result
        self.expr_cache[frozen_specs] = lambda_expr
        return lambda_expr

    def __call__(self, *args, **kwargs):
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_trees = []
        arg_specs = []
        arg_masks = []
        for name, arg in bound_args.arguments.items():
            is_static = name in self.static_argnames
            arg_masks.append(not is_static)
            if is_static:
                arg_specs.append(arg)
            else:
                node = pt.from_value(arg)
                spec = ArgSpec.from_pytree(node)
                arg_trees.append(node)
                arg_specs.append(spec)
        lambda_expr = self.get_expr(*arg_specs, pred=lambda i, _: arg_masks[i])
        for transform in self.transforms:
            lambda_expr = transform(lambda_expr)
        return Interpreter(tuple(arg_trees))(lambda_expr)

class grad(Transform):
    def __init__(self, fn, *, static_argnames=None, wrt=None):
        super().__init__(fn, transforms=(
            Gradient(wrt_args=wrt),
            CommonSubexpressionElimination(),
            ConstantFolding(),
            # AlgebraicSimplification(),
            DeadCodeElimination()
        ), static_argnames=static_argnames)

class jit(Transform):
    def __init__(self, fn, *, static_argnames=None):
        super().__init__(fn, static_argnames=static_argnames)

class Gradient:
    def __init__(self, gradient_only=True, wrt_args=None):
        self.equation_map: list[Tracer] = []
        self.supervisor = TracerSupervisor()
        self.derivatives: defaultdict = defaultdict(lambda: _ensure_tracer(0.0, self.supervisor))
        self.gradient_only = gradient_only
        self.wrt_args = wrt_args

    def _handle_broadcast_derivative(self, input_idx: int, input_shape: Tuple[int, ...],
                                   output_shape: Tuple[int, ...], derivative: Tracer) -> None:
        """Handle summing over broadcasted dimensions for elementwise operations"""
        # Get number of leading dimensions from derivative shape if in jacobian case
        n_leading = len(derivative.shape) - len(output_shape)

        # Build list of axes to sum over
        sum_axes = []

        padded_shape = []
        for i in range(len(input_shape), len(output_shape)):
            padded_shape.append(1)
        padded_shape.extend(input_shape)

        # Check dimensions that were broadcast from 1 to match output
        for i, (s1, s2) in enumerate(zip(padded_shape, output_shape)):
            if s1 == 1 and s2 > 1:
                sum_axes.append(i + n_leading)

        # Sum over broadcast dimensions if any
        result = derivative.sum(tuple(sum_axes), keepdims=True) if sum_axes else derivative

        # Reshape to match input shape while preserving leading dimensions
        new_shape = derivative.shape[:n_leading] + input_shape
        result = result.reshape(*new_shape)
        self.derivatives[input_idx] += result

    def visit_literal(self, eq: Literal, derivative: Tracer) -> None:
        # Always 0, nothing upstream of it
        pass

    def visit_var(self, eq: Var, derivative: Tracer) -> None:
        # Always 1, nothing upstream of it
        pass

    def visit_add(self, eq: Op, derivative: Tracer) -> None:
        for input_idx in eq.inputs:
            if isinstance(derivative, pt.Node):
                derivative = derivative.leaf_value
            # For addition, derivative wrt each input is just the derivative
            self.derivatives[input_idx] = _ensure_tracer(derivative, self.supervisor)

    def visit_mul(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        self.derivatives[eq.inputs[0]] += derivative * b
        self.derivatives[eq.inputs[1]] += derivative * a

    def visit_sub(self, eq: Op, derivative: Tracer) -> None:
        self.derivatives[eq.inputs[0]] += derivative
        self.derivatives[eq.inputs[1]] += -derivative

    def visit_div(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        self.derivatives[eq.inputs[0]] += derivative / b
        self.derivatives[eq.inputs[1]] += -derivative * a / (b * b)

    def visit_pow(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        self.derivatives[eq.inputs[0]] += derivative * b * a ** (b - 1)
        self.derivatives[eq.inputs[1]] += derivative * (a ** b) * log(a)

    def visit_neg(self, eq: Op, derivative: Tracer) -> None:
        self.derivatives[eq.inputs[0]] += -derivative

    def visit_exp(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self.derivatives[eq.inputs[0]] += derivative * exp(input_val)

    def visit_log(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self.derivatives[eq.inputs[0]] += derivative / input_val

    def visit_sin(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self.derivatives[eq.inputs[0]] += derivative * cos(input_val)

    def visit_cos(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self.derivatives[eq.inputs[0]] += -derivative * sin(input_val)

    def visit_abs(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self.derivatives[eq.inputs[0]] += derivative * abs(input_val) / input_val

    def visit_maximum(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        mask_a = (a > b)
        mask_b = (b > a)
        equal = (a == b)
        self.derivatives[eq.inputs[0]] += derivative * (mask_a + equal * 0.5)
        self.derivatives[eq.inputs[1]] += derivative * (mask_b + equal * 0.5)

    def visit_minimum(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        mask_a = (a < b)
        mask_b = (b < a)
        equal = (a == b)
        self.derivatives[eq.inputs[0]] += derivative * (mask_a + equal * 0.5)
        self.derivatives[eq.inputs[1]] += derivative * (mask_b + equal * 0.5)

    def visit_dot(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]

        # Extract dimensions from metadata
        lhs_c = eq.metadata['lhs_contracting_dims']
        rhs_c = eq.metadata['rhs_contracting_dims']
        lhs_b = eq.metadata['lhs_batch_dims']
        rhs_b = eq.metadata['rhs_batch_dims']

        # For jacobian case, need to preserve leading dimensions from derivative
        if len(derivative.shape) > len(eq.shape):
            leading_dims = derivative.shape[:-len(eq.shape)]
        else:
            leading_dims = ()

        # Handle derivative wrt first arg (a)
        # Need to contract derivative with b in appropriate dimensions
        # C_ji = sum(k, A_ki * B_kj)
        # dL/d_Aki = sum(j, dL/dC_ji * dC_ji/d_Aki)
        # dC_ji/d_Aki = B_kj (zero everywhere else)
        # dL/d_Aki = sum(j, dL/dC_ji * B_kj) # contract shared non-contracting dims
        # dL/d_Bkj = sum(i, dL/dC_ji * dC_ji/d_Bkj)
        # dC_ji/d_Bkj = A_ki (zero elsewhere)
        # dL/d_Bkj = sum(i, dL/dC_ji * A_ki) # contract shared non-contracting dims

        # For derivative wrt a:
        # Need to match derivative dims [*leading, *batch, *a_free, *b_free] with b's dims [*batch, *b_free, *contract]
        # to get result dims [*leading, *batch, *a_free, *contract]
        b_free_dims = [i for i in range(len(b.shape)) if i not in rhs_c and i not in rhs_b]
        a_free_dims = [i for i in range(len(a.shape)) if i not in lhs_c and i not in lhs_b]
        derivative_b_free_offset = len(leading_dims) + len(lhs_b) + len(a_free_dims)
        da = derivative.dot_general(b,
            lhs_contracting_dims=tuple(range(derivative_b_free_offset, derivative_b_free_offset + len(b_free_dims))),
            rhs_contracting_dims=tuple(b_free_dims),
            lhs_batch_dims=tuple(range(len(leading_dims), len(leading_dims) + len(lhs_b))),
            rhs_batch_dims=rhs_b)
        assert da.shape == leading_dims + a.shape
        self.derivatives[eq.inputs[0]] += da

        # Handle derivative wrt second arg (b)
        # Need to match derivative dims [*leading, *batch, *a_free, *b_free] with a's dims [*batch, *a_free, *contract]
        # to get result dims [*leading, *batch, *b_free, *contract]
        derivative_a_free_offset = len(leading_dims) + len(lhs_b) + len(b_free_dims)
        db = derivative.dot_general(a,
            lhs_contracting_dims=tuple(range(len(leading_dims) + len(lhs_b), derivative_a_free_offset)),
            rhs_contracting_dims=tuple(a_free_dims),
            lhs_batch_dims=tuple(range(len(leading_dims), len(leading_dims) + len(lhs_b))),
            rhs_batch_dims=lhs_b)
        assert db.shape == leading_dims + b.shape
        self.derivatives[eq.inputs[1]] += db


    def visit_transpose(self, eq: Op, derivative: Tracer) -> None:
        axes = eq.metadata['axes']
        # Compute inverse permutation
        inverse_axes = [0] * len(axes)
        for i, axis in enumerate(axes):
            inverse_axes[axis] = i
        # For Jacobian case, preserve leading dims and append inverse permutation
        if len(derivative.shape) > len(eq.shape):
            leading_dims = derivative.shape[:-len(eq.shape)]
            inverse_axes = tuple(range(len(leading_dims))) + tuple(i + len(leading_dims) for i in inverse_axes)
        self.derivatives[eq.inputs[0]] += derivative.transpose(*inverse_axes)


    def visit_reshape(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        if len(derivative.shape) > len(eq.shape):
            # Jacobian case - preserve leading dims and reshape output dims
            leading_dims = derivative.shape[:-len(eq.shape)]
            new_shape = leading_dims + input_val.shape
            self.derivatives[eq.inputs[0]] += derivative.reshape(*new_shape)
        else:
            # Gradient case - just reshape to match input
            self.derivatives[eq.inputs[0]] += derivative.reshape(*input_val.shape)

    def visit_broadcast(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        self._handle_broadcast_derivative(eq.inputs[0], input_val.shape, eq.shape, derivative)

    def visit_sum(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        input_eq = input_val.supervisor.equations[input_val.idx]
        output_shape = input_eq.shape

        # For jacobian case, we need to preserve leading dims from derivative
        # and append original input shape for broadcasting
        broadcast_shape = derivative.shape[:-len(eq.shape)] + output_shape

        self.derivatives[eq.inputs[0]] += derivative.broadcast_to(broadcast_shape)

    def visit_prod(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        axes = eq.metadata['axes']
        reduced = prod(input_val, axis=axes, keepdims=True)
        self.derivatives[eq.inputs[0]] += derivative * reduced / input_val

    def visit_min(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        axes = eq.metadata['axes']
        mask = (input_val == min(input_val, axis=axes, keepdims=True))
        self.derivatives[eq.inputs[0]] += derivative * mask

    def visit_max(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        axes = eq.metadata['axes']
        mask = (input_val == max(input_val, axis=axes, keepdims=True))
        self.derivatives[eq.inputs[0]] += derivative * mask

    def visit_op(self, eq: Op, derivative: Tracer) -> None:
        method = f'visit_{eq.op.value}'
        visitor = getattr(self, method)
        visitor(eq, derivative)

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Generates gradient/jacobian of lambda expression with respect to inputs"""
        # Verify shape based on gradient_only
        if self.gradient_only and not lambda_expr.result.leaf_value is not None:
            raise ValueError("gradient_only=True requires scalar-valued functions")

        # Get result node(s)
        results = []
        print("\nGradient.__call__ debug:")  # Debug print
        print(f"Initial lambda_expr.result: {lambda_expr.result}")  # Debug print
        pt.map(lambda idx: results.append(idx), lambda_expr.result)
        print(f"Collected results: {results}")  # Debug print

        # Copy over original equations
        self.equation_map = [Tracer(eq, self.supervisor) for eq in lambda_expr.equations]
        print(f"Number of equations: {len(lambda_expr.equations)}")  # Debug print
        for i, eq in enumerate(lambda_expr.equations):
            print(f"Equation {i}: {eq}")  # Debug print

        # Filter args if specified
        if self.wrt_args is not None:
            active_args = [i in self.wrt_args for i in range(len(lambda_expr.args))]
        else:
            active_args = [True] * len(lambda_expr.args)

        # Compute derivatives for each output
        all_derivatives = []
        for result_idx in results:
            print(f"\nProcessing result_idx: {result_idx}")  # Debug print
            # reinit derivatives for this output
            self.derivatives = defaultdict(lambda: _ensure_tracer(0.0, self.supervisor))

            # Initialize derivative of result wrt result as identity matrix
            result_eq = lambda_expr.equations[result_idx]
            print(f"Creating identity jacobian for result_idx {result_idx}, shape {result_eq.shape}")  # Debug print
            # For a tensor of shape (n,), we need an identity matrix of shape (n,n)
            # Reshape to match output dimensions for proper broadcasting
            identity = np.eye(np.prod(result_eq.shape)).reshape(result_eq.shape + result_eq.shape)
            self.derivatives[result_idx] = _ensure_tracer(identity, self.supervisor)
            print(f"Initial derivatives: {dict(self.derivatives)}")  # Debug print

            # Work backwards through equations propagating derivatives
            for i in range(len(lambda_expr.equations)-1, -1, -1):
                if i not in self.derivatives:
                    continue
                eq = lambda_expr.equations[i]
                derivative = self.derivatives[i]

                method = f'visit_{type(eq).__name__.lower()}'
                visitor = getattr(self, method)
                visitor(eq, derivative)

            # Collect derivatives for this output
            def to_grad(idx):
                if isinstance(idx, pt.Node):
                    idx = idx.leaf_value
                if idx not in self.derivatives:
                    return _ensure_tracer(np.zeros(lambda_expr.equations[idx].shape + result_eq.shape), self.supervisor)
                derivative = self.derivatives[idx]
                return derivative
            derivatives = tuple(pt.map(to_grad, arg) for i, arg in enumerate(lambda_expr.args)
                             if active_args[i])
            print(f"Derivatives for result {result_idx}: {derivatives}")  # Debug print
            all_derivatives.append(derivatives)

        # Create result based on gradient_only
        if self.gradient_only:
            assert len(all_derivatives) == 1
            result = all_derivatives[0]
            assert isinstance(result, tuple)
            # Pack into tuple/sequence if needed
            if len(lambda_expr.args) == 1:
                result = result[0]
            else:
                result = pt.from_sequence(tuple(result))
        else:
            # Create tree of jacobians
            def make_jacobian_node(result_idx, derivatives, output_results):
                # Find which output this result corresponds to
                out_node = lambda_expr.result
                if isinstance(out_node, pt.Node):
                    output_idx = output_results.index(out_node.leaf_value)
                else:
                    output_idx = output_results.index(result_idx)

                # Get the result for each input
                def get_result(idx):
                    if isinstance(idx, pt.Node):
                        idx = idx.leaf_value
                    if isinstance(idx, tuple):
                        return tuple(get_result(i) for i in idx)
                    if isinstance(idx, Tracer):
                        return idx
                    # If idx is not in derivatives, return zero matrix of appropriate shape
                    if idx not in self.derivatives:
                        result_shape = lambda_expr.equations[result_idx].shape
                        input_shape = lambda_expr.equations[idx].shape if idx < len(lambda_expr.equations) else (1,)
                        zero_shape = input_shape + result_shape
                        return _ensure_tracer(np.zeros(zero_shape), self.supervisor)
                    return self.derivatives[idx]

                derivative_results = tuple(get_result(d) for d in derivatives)
                # Convert results to PyTree nodes
                final_results = tuple(pt.leaf(r) if isinstance(r, Tracer) else r for r in derivative_results)
                return pt.from_sequence(final_results)

            result = [make_jacobian_node(result_idx, derivatives, results)
                      for result_idx, derivatives in zip(results, all_derivatives)]
            # Convert result list to PyTree if needed
            if len(result) == 1:
                result = result[0]
                # Ensure result is a PyTree node
                if not isinstance(result, pt.Node):
                    result = pt.from_sequence(result)
            else:
                result = pt.from_sequence(result)

        return self.supervisor.create_lambda(lambda_expr.args, result)

class DeadCodeElimination:
    def __init__(self):
        self.used_equations = set()

    def mark_used(self, result: Union[int, pt.Node, tuple]):
        """Recursively mark all equations needed to compute result"""
        if isinstance(result, tuple):
            # Handle tuple by recursively marking each element
            for x in result:
                self.mark_used(x)
            return
        elif isinstance(result, int):
            # Base case - mark this equation
            if result not in self.used_equations:
                self.used_equations.add(result)
                # Recursively mark inputs
                eq = self.equations[result]
                if isinstance(eq, Op):
                    # Always include IO callbacks even if otherwise unused
                    if eq.op == OpType.IO_CALLBACK:
                        self.used_equations.add(result)
                    for input_idx in eq.inputs:
                        self.mark_used(input_idx)
                elif isinstance(eq, Var):
                    # Variables are leaves
                    pass
                elif isinstance(eq, Literal):
                    # Literals are leaves
                    pass
                else:
                    raise ValueError(f"Unknown equation type: {type(eq)}")
        elif isinstance(result, pt.Node):
            # Result is a PyTree - recursively process leaves
            if result.leaf_value is not None:
                self.mark_used(result.leaf_value)
            else:
                for field in result.fields.values():
                    self.mark_used(field)

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Eliminates dead code from lambda expression"""
        self.equations = lambda_expr.equations
        self.used_equations = set()

        # Mark all equations needed for result
        self.mark_used(lambda_expr.result)

        # Add all IO callback ops and their dependencies even if otherwise unused
        for i, eq in enumerate(lambda_expr.equations):
            if isinstance(eq, Op) and eq.op == OpType.IO_CALLBACK:
                self.mark_used(i)

        # Create new equations list with only used equations
        new_equations = []
        old_to_new = {}
        for i, eq in enumerate(lambda_expr.equations):
            if i in self.used_equations:
                # Update indices in Op nodes
                if isinstance(eq, Op):
                    new_inputs = [old_to_new[idx] for idx in eq.inputs]
                    eq = Op(eq.op, new_inputs, eq.dtype, eq.shape, eq.metadata)
                new_equations.append(eq)
                old_to_new[i] = len(new_equations) - 1

        # Update result indices
        def update_index(idx):
            if isinstance(idx, tuple):
                # Handle tuple of Nodes
                idx = tuple(n.leaf_value if isinstance(n, pt.Node) else n for n in idx)
            elif isinstance(idx, pt.Node):
                # Handle single Node
                idx = idx.leaf_value
            return old_to_new.get(idx, idx)
        assert isinstance(lambda_expr.result, pt.Node)
        new_result = pt.map(update_index, lambda_expr.result)
        assert isinstance(new_result, pt.Node)
        return Lambda(lambda_expr.args, new_equations, new_result)

class CommonSubexpressionElimination:
    def __init__(self):
        self.expr_to_idx = {}

    def get_expr_key(self, eq: Expr) -> tuple:
        if isinstance(eq, Op):
            return (eq.op, tuple(eq.inputs), pt.freeze(eq.metadata))
        elif isinstance(eq, Literal):
            return ('literal', eq.value.tobytes())
        elif isinstance(eq, Var):
            return ('var', eq.arg_index)
        else:
            raise ValueError(f"Unknown equation type: {type(eq)}")

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Eliminates common subexpressions from lambda expression"""
        # Reset state each call
        self.expr_to_idx = {}

        new_equations = []
        old_to_new = {}

        for i, eq in enumerate(lambda_expr.equations):
            key = self.get_expr_key(eq)
            if key in self.expr_to_idx:
                # Reuse existing computation
                old_to_new[i] = self.expr_to_idx[key]
                continue

            # Add new unique computation
            if isinstance(eq, Op):
                new_inputs = [old_to_new[idx] for idx in eq.inputs]
                eq = Op(eq.op, new_inputs, eq.dtype, eq.shape, eq.metadata)
            new_equations.append(eq)
            old_to_new[i] = len(new_equations) - 1
            self.expr_to_idx[key] = old_to_new[i]

        # Update result indices
        def update_index(idx):
            if isinstance(idx, tuple):
                # Handle tuple of Nodes by recursively handling each element
                return tuple(update_index(x) for x in idx)
            elif isinstance(idx, pt.Node):
                # Handle single Node
                idx = idx.leaf_value
            return old_to_new.get(idx, idx)
        assert isinstance(lambda_expr.result, pt.Node)
        new_result = pt.map(update_index, lambda_expr.result)
        assert isinstance(new_result, pt.Node)
        return Lambda(lambda_expr.args, new_equations, new_result)

class ConstantFolding:
    def __init__(self):
        self.constants = {}

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Evaluates constant expressions and replaces them with literals"""
        # Reset state
        self.constants = {}

        new_equations = []
        old_to_new = {}

        for i, eq in enumerate(lambda_expr.equations):
            if isinstance(eq, Literal):
                # Literals are already constant
                new_equations.append(eq)
                old_to_new[i] = len(new_equations) - 1
                self.constants[i] = eq.value
                continue

            elif isinstance(eq, Var):
                # Variables are not constant
                new_equations.append(eq)
                old_to_new[i] = len(new_equations) - 1
                continue

            elif isinstance(eq, Op):
                # Check if all inputs are constant
                all_constant = all(i in self.constants for i in eq.inputs)

                if all_constant:
                    # Can evaluate this expression
                    # Create minimal interpreter for just this op
                    input_values = tuple(self.constants.get(idx) for idx in eq.inputs)

                    # Skip if any input values are None
                    if any(v is None for v in input_values):
                        new_equations.append(eq)
                        old_to_new[i] = len(new_equations) - 1
                        continue

                    supervisor = TracerSupervisor()
                    interpreter = Interpreter(input_values)
                    inputs = []
                    for input in input_values:
                        inputs.append(supervisor.add_equation(Literal(input)))
                    result = supervisor.add_equation(Op(eq.op, inputs, eq.dtype, eq.shape))
                    lam = supervisor.create_lambda((), pt.leaf(result))
                    value = interpreter(lam)
                    literal = Literal(value)
                    new_equations.append(literal)
                    old_to_new[i] = len(new_equations) - 1
                    self.constants[i] = value
                    continue

            # Not constant - update input indices and add to new equations
            if isinstance(eq, Op):
                new_inputs = [old_to_new.get(idx, idx) for idx in eq.inputs]
                eq = Op(eq.op, new_inputs, eq.dtype, eq.shape, eq.metadata)
            new_equations.append(eq)
            old_to_new[i] = len(new_equations) - 1

        # Update result indices
        def update_index(idx):
            if isinstance(idx, tuple):
                # Handle tuple of Nodes by recursively handling each element
                return tuple(update_index(x) for x in idx)
            elif isinstance(idx, pt.Node):
                # Handle single Node
                idx = idx.leaf_value
            return old_to_new.get(idx, idx)
        assert isinstance(lambda_expr.result, pt.Node)
        new_result = pt.map(update_index, lambda_expr.result)
        assert isinstance(new_result, pt.Node)

        return Lambda(lambda_expr.args, new_equations, new_result)

class AlgebraicSimplification:
    def __init__(self):
        self.new_equations = []
        self.old_to_new = {}

    def can_simplify(self, eq: Expr) -> bool:
        if not isinstance(eq, Op):
            return False
        if eq.op != OpType.ADD:
            return False
        in1, in2 = [self.new_equations[self.old_to_new[idx]] for idx in eq.inputs]
        if (isinstance(in1, Literal) and np.all(in1.value == 0.0)) or \
           (isinstance(in2, Literal) and np.all(in2.value == 0.0)):
            return True
        return False

    def simplify(self, eq: Op) -> int:
        in1, in2 = [self.new_equations[self.old_to_new[idx]] for idx in eq.inputs]
        if isinstance(in1, Literal) and np.all(in1.value == 0.0):
            return self.old_to_new[eq.inputs[1]]
        else:
            return self.old_to_new[eq.inputs[0]]

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Simplifies algebraic expressions"""
        self.new_equations = []
        self.old_to_new = {}

        assert isinstance(lambda_expr.result, pt.Node)
        for i, eq in enumerate(lambda_expr.equations):
            if not self.can_simplify(eq):
                # Keep equation as-is, just update input indices for Ops
                if isinstance(eq, Op):
                    new_inputs = [self.old_to_new[idx] for idx in eq.inputs]
                    eq = Op(eq.op, new_inputs, eq.dtype, eq.shape, eq.metadata)
                self.new_equations.append(eq)
                self.old_to_new[i] = len(self.new_equations) - 1
            else:
                # Replace with simplified version
                assert isinstance(eq, Op)
                self.old_to_new[i] = self.simplify(eq)

        # Update result indices
        def update_index(idx):
            if isinstance(idx, tuple):
                # Handle tuple of Nodes by recursively handling each element
                return tuple(update_index(x) for x in idx)
            elif isinstance(idx, pt.Node):
                # Handle single Node
                idx = idx.leaf_value
            return self.old_to_new.get(idx, idx)
        new_result = pt.map(update_index, lambda_expr.result)
        assert isinstance(new_result, pt.Node)

        return Lambda(lambda_expr.args, self.new_equations, new_result)

def transform_pipeline(*transforms):
    def apply_transforms(lmd):
        result = lmd
        for t in transforms:
            result = t(result)
        return result
    return apply_transforms

if __name__ == '__main__':
    from dataclasses import dataclass

    @dataclass
    class Model:
        weights: np.ndarray
        bias: np.ndarray

    @jit
    def loss(model, inputs):
        # Compute neural network output
        hidden = inputs @ model.weights + model.bias
        output = sin(hidden)
        return sum(output**2)

    # Create model and inputs
    model = Model(
        weights=np.array([[1., 2.], [3., 4.]]),
        bias=np.array([0.1, 0.2])
    )
    inputs = np.array([[0.5, 0.6]])

    # Get gradient of loss w.r.t. model params
    loss_grad = grad(loss)

    (dmodel, dinputs) = loss_grad(model, inputs)

    print("dweights =", dmodel.weights)
    print("dbias =", dmodel.bias)
