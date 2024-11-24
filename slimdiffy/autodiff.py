# Copyright (c) 2024 Jake Ehrlich
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # type: ignore
from collections import defaultdict
import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
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
    MATMUL = 'matmul'
    TRANSPOSE = 'transpose'

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
        return Lambda(args, self.equations, result)

class Tracer:
    def __init__(self, expr: Expr, supervisor: TracerSupervisor):
        self.supervisor = supervisor
        self.idx = supervisor.add_equation(expr)

    def _binary_op(self, other, op: OpType) -> 'Tracer':
        other = _ensure_tracer(other, self.supervisor)
        self_expr = self.supervisor.equations[self.idx]
        other_expr = self.supervisor.equations[other.idx]

        try:
            out_shape = np.broadcast_shapes(self_expr.shape, other_expr.shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shapes {self_expr.shape} and {other_expr.shape}")

        return Tracer(Op(op, [self.idx, other.idx], self_expr.dtype, out_shape), self.supervisor)

    def _unary_op(self, op: OpType) -> 'Tracer':
        self_expr = self.supervisor.equations[self.idx]
        return Tracer(Op(op, [self.idx], self_expr.dtype, self_expr.shape), self.supervisor)

    def broadcast_to(self, shape: Tuple[int, ...]) -> 'Tracer':
        self_expr = self.supervisor.equations[self.idx]
        try:
            np.broadcast_shapes(self_expr.shape, shape)
        except ValueError:
            raise ValueError(f"Cannot broadcast shape {self_expr.shape} to {shape}")
        return Tracer(Op(OpType.BROADCAST, [self.idx], self_expr.dtype, shape), self.supervisor)

    def __add__(self, other):
        return self._binary_op(other, OpType.ADD)

    def __radd__(self, other):
        return self._binary_op(other, OpType.ADD)

    def __mul__(self, other):
        return self._binary_op(other, OpType.MUL)

    def __rmul__(self, other):
        return self._binary_op(other, OpType.MUL)

    def __truediv__(self, other):
        return self._binary_op(other, OpType.DIV)

    def __rtruediv__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        return other._binary_op(self, OpType.DIV)

    def __matmul__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        self_expr = self.supervisor.equations[self.idx]
        other_expr = self.supervisor.equations[other.idx]

        if len(self_expr.shape) != 2 or len(other_expr.shape) != 2:
            raise ValueError("matmul requires 2D arrays")
        if self_expr.shape[1] != other_expr.shape[0]:
            raise ValueError(f"Cannot multiply arrays of shapes {self_expr.shape} and {other_expr.shape}")
        out_shape = (self_expr.shape[0], other_expr.shape[1])

        return Tracer(Op(OpType.MATMUL, [self.idx, other.idx], self_expr.dtype, out_shape), self.supervisor)

    def __rmatmul__(self, other):
        other = _ensure_tracer(other, self.supervisor)
        return other.__matmul__(self)

    def __sub__(self, other):
        return self._binary_op(other, OpType.SUB)

    def __rsub__(self, other):
        return self._binary_op(other, OpType.SUB)

    def __pow__(self, other):
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

    @property
    def T(self):
        self_expr = self.supervisor.equations[self.idx]
        if len(self_expr.shape) != 2:
            raise ValueError("transpose requires 2D array")
        return Tracer(Op(OpType.TRANSPOSE, [self.idx], self_expr.dtype,
                        (self_expr.shape[1], self_expr.shape[0])), self.supervisor)

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

        # Use pytree to construct result
        def get_result(idx: int) -> Any:
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

    def visit_op_matmul(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]] @ self.results[expr.inputs[1]]

    def visit_op_transpose(self, expr: Op) -> Any:
        return self.results[expr.inputs[0]].T

    def visit_op_broadcast(self, expr: Op) -> Any:
        x = np.asarray(self.results[expr.inputs[0]])
        return np.broadcast_to(x, expr.shape)

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
    def __init__(self, fn, transforms=()):
        # If fn is already a Transform, compose the transforms
        if isinstance(fn, Transform):
            self.fn = fn.fn
            self.signature = fn.signature
            self.transforms = fn.transforms + transforms
        else:
            self.fn = fn
            self.signature = inspect.signature(fn)
            self.transforms = transforms

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

    def get_expr(self, *arg_specs):
        """Get expression graph for function given input specs"""
        supervisor = TracerSupervisor()
        full_tree = pt.from_sequence(arg_specs)
        index_tree, = self.index_pytrees(full_tree)
        def make_var(arg_spec, arg_index):
            return Tracer(Var(arg_index, arg_spec.dtype, arg_spec.shape), supervisor)
        trace_vars = pt.map(make_var, full_tree, index_tree).to_value()
        result = self.fn(*trace_vars)
        index_tuple = index_tree.to_sequence()
        assert isinstance(index_tuple, Tuple)
        if isinstance(result, Tracer):
            return supervisor.create_lambda(index_tuple, pt.leaf(result.idx))
        pytree = pt.from_value(result)
        pytree = pt.map(lambda x: x.idx, pytree)
        return supervisor.create_lambda(index_tuple, pytree)

    def __call__(self, *args, **kwargs):
        bound_args = self.signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        arg_trees = []
        arg_specs = []
        for arg in bound_args.arguments.values():
            node = pt.from_value(arg)
            spec = ArgSpec.from_pytree(node)
            arg_trees.append(node)
            arg_specs.append(spec)
        lambda_expr = self.get_expr(*arg_specs)
        for transform in self.transforms:
            lambda_expr = transform(lambda_expr)
        return Interpreter(tuple(arg_trees))(lambda_expr)

class grad(Transform):
    def __init__(self, fn):
        super().__init__(fn, transforms=(
            Gradient(),
            CommonSubexpressionElimination(),
            ConstantFolding(),
            DeadCodeElimination()
        ))

class jit(Transform):
    def __init__(self, fn):
        super().__init__(fn)

    def transform(self, transform_fn, *arg_specs):
        """Helper for applying arbitrary transforms"""
        lambda_expr = self.get_expr(*arg_specs)
        return transform_fn(lambda_expr)

class Gradient:
    def __init__(self):
        self.equation_map: list[Tracer] = []
        self.supervisor = TracerSupervisor()
        self.derivatives: defaultdict = defaultdict(lambda: _ensure_tracer(0.0, self.supervisor))

    def visit_literal(self, eq: Literal, derivative: Tracer) -> None:
        # Always 0, nothing upstream of it
        pass

    def visit_var(self, eq: Var, derivative: Tracer) -> None:
        # Always 1, nothing upstream of it
        pass

    def visit_add(self, eq: Op, derivative: Tracer) -> None:
        for input_idx in eq.inputs:
            self.derivatives[input_idx] += derivative

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

    def visit_matmul(self, eq: Op, derivative: Tracer) -> None:
        a, b = [self.equation_map[idx] for idx in eq.inputs]
        self.derivatives[eq.inputs[0]] += derivative @ b.T
        self.derivatives[eq.inputs[1]] += a.T @ derivative

    def visit_transpose(self, eq: Op, derivative: Tracer) -> None:
        self.derivatives[eq.inputs[0]] += derivative.T

    def visit_sum(self, eq: Op, derivative: Tracer) -> None:
        input_val = self.equation_map[eq.inputs[0]]
        output_shape = input_val.supervisor.equations[input_val.idx].shape
        self.derivatives[eq.inputs[0]] += derivative.broadcast_to(output_shape)

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
        """Generates gradient of lambda expression with respect to inputs"""
        # Verify result is a scalar
        if not (isinstance(lambda_expr.result, pt.Node) and lambda_expr.result.leaf_value is not None):
            raise ValueError("Can only take gradient of scalar-valued functions")

        result = lambda_expr.result.leaf_value

        # reinit supervisor and derivatives
        self.supervisor = TracerSupervisor()
        self.derivatives = defaultdict(lambda: _ensure_tracer(0.0, self.supervisor))

        # Copy over original equations
        self.equation_map = [Tracer(eq, self.supervisor) for eq in lambda_expr.equations]

        # Initialize derivative of result wrt result as 1.0
        self.derivatives[result] = _ensure_tracer(1.0, self.supervisor)

        # Work backwards through equations propagating derivatives
        for i in range(len(lambda_expr.equations)-1, -1, -1):
            if i not in self.derivatives:
                continue
            eq = lambda_expr.equations[i]
            derivative = self.derivatives[i]

            if isinstance(eq, Op):
                self.visit_op(eq, derivative)
            else:
                method = f'visit_{type(eq).__name__.lower()}'
                visitor = getattr(self, method)
                visitor(eq, derivative)

        def to_grad(idx):
            return self.derivatives[idx].idx
        result = tuple(pt.map(to_grad, arg) for arg in lambda_expr.args)

        # Return lambda that computes all input derivatives
        if len(result) == 1:
            result = result[0]
        else:
            result = pt.from_sequence(result)
        return self.supervisor.create_lambda(lambda_expr.args, result)

class DeadCodeElimination:
    def __init__(self):
        self.used_equations = set()

    def mark_used(self, result: Union[int, pt.Node]):
        """Recursively mark all equations needed to compute result"""
        if isinstance(result, int):
            # Base case - mark this equation
            if result not in self.used_equations:
                self.used_equations.add(result)
                # Recursively mark inputs
                eq = self.equations[result]
                if isinstance(eq, Op):
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
        else:
            # Result is a PyTree - recursively process leaves
            if result.leaf_value is not None:
                self.mark_used(result.leaf_value)
            else:
                for field in result.fields.values():
                    self.mark_used(field)

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Eliminates dead code from lambda expression"""
        self.equations = lambda_expr.equations

        # Mark all equations needed for result
        self.mark_used(lambda_expr.result)

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
        def update_index(idx: int) -> int:
            return old_to_new.get(idx, idx)
        new_result = pt.map(update_index, lambda_expr.result)

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
        def update_index(idx: int) -> int:
            return old_to_new[idx]
        new_result = pt.map(update_index, lambda_expr.result)

        return Lambda(lambda_expr.args, new_equations, new_result)

class ConstantFolding:
    def __init__(self):
        self.constants = {}
        self.interpreter = None

    def __call__(self, lambda_expr: Lambda) -> Lambda:
        """Evaluates constant expressions and replaces them with literals"""
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
                    input_values = tuple(self.constants[idx] for idx in eq.inputs)

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
                new_inputs = [old_to_new[idx] for idx in eq.inputs]
                eq = Op(eq.op, new_inputs, eq.dtype, eq.shape, eq.metadata)
            new_equations.append(eq)
            old_to_new[i] = len(new_equations) - 1

        # Update result indices
        def update_index(idx: int) -> int:
            return old_to_new[idx]
        new_result = pt.map(update_index, lambda_expr.result)

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
        def update_index(idx: int) -> int:
            return self.old_to_new[idx]
        new_result = pt.map(update_index, lambda_expr.result)

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
