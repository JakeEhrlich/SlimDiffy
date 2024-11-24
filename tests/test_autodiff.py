# Copyright (c) 2024 Jake Ehrlich
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # type: ignore
import slimdiffy.autodiff as ad
import slimdiffy.pytree as pt
import dataclasses

def test_from_to_value():
    # Test various python objects round-trip through pytree
    test_cases = [
        1,
        1.5,
        [1, 2, 3],
        (4, 5, 6),
        {'a': 1, 'b': 2},
        {'x': [1, 2], 'y': {'z': 3}},
        [{'a': [1, 2]}, (3, {'b': 4})],
        ExampleDataClass(y={'test': 5}, z=[7,8]),
        np.array([1, 2, 3])
    ]

    for test_case in test_cases:
        # Convert to pytree and back
        node = pt.from_value(test_case)
        result = node.to_value()

        # Handle numpy arrays specially
        if isinstance(test_case, np.ndarray):
            assert isinstance(result, np.ndarray)
            assert np.array_equal(result, test_case)
            continue

        # Handle dataclasses specially
        if dataclasses.is_dataclass(test_case):
            assert type(result) == type(test_case)
            assert result.__dict__ == test_case.__dict__
            continue

        # For other types, direct equality should work
        assert result == test_case


def test_render():
    x = [[1, 2], {'a': 3, 'b': {'c': 4}}, (5, 6)]
    node = pt.from_value(x)

    rendered = node.render()
    expected = """[
  [
    1,
    2
  ],
  {
    'a': 3,
    'b': {
      'c': 4
    }
  },
  (
    5,
    6
  )
]"""
    assert rendered == expected

    # Test custom render function
    def custom_render(x):
        return f"<{x}>"

    rendered = node.render(render_leaf=custom_render)
    expected = """[
  [
    <1>,
    <2>
  ],
  {
    'a': <3>,
    'b': {
      'c': <4>
    }
  },
  (
    <5>,
    <6>
  )
]"""
    assert rendered == expected

    # Test rendering dataclass
    @dataclasses.dataclass
    class TestClass:
        a: int
        b: str

    x = TestClass(a=1, b="test")
    node = pt.from_value(x)

    rendered = node.render()
    expected = """TestClass(
  a=1,
  b='test'
)"""
    assert rendered == expected

    rendered = node.render(render_leaf=custom_render)
    expected = """TestClass(
  a=<1>,
  b=<test>
)"""
    assert rendered == expected


@dataclasses.dataclass
class ExampleDataClass:
    x: int = pt.static_field(1)
    y: dict = dataclasses.field(default_factory=dict)
    z: list = dataclasses.field(default_factory=list)

def test_dataclass():
    obj = ExampleDataClass(y={'a': 1}, z=[1,2])
    node = pt.from_value(obj)

    # Test static field 'x' goes into metadata
    assert node.metadata['x'] == 1

    # Test non-static fields are in fields dict
    assert set(node.fields.keys()) == {'y', 'z'}

    # Test nested dict/list structure
    assert node.fields['y'].typ is dict
    assert node.fields['y'].fields['a'].leaf_value == 1
    assert node.fields['z'].typ is list
    assert node.fields['z'].fields[0].leaf_value == 1
    assert node.fields['z'].fields[1].leaf_value == 2

    # Test round-trip
    obj2 = node.to_value()
    assert obj2.x == obj.x
    assert obj2.y == obj.y
    assert obj2.z == obj.z

def test_mapkeys():
    x = [1, 2, {'a': 3, 'b': 4}]
    node = pt.from_value(x)

    def print_path(path, value):
        assert type(value) == int
        return f"{'.'.join(str(p) for p in path)}={value}"

    result = pt.mapkeys(print_path, node)
    assert result.fields[0].leaf_value == "0=1"
    assert result.fields[1].leaf_value == "1=2"
    assert result.fields[2].fields['a'].leaf_value == "2.a=3"
    assert result.fields[2].fields['b'].leaf_value == "2.b=4"

def test_map():
    x = [1, 2, {'a': 3, 'b': 4}]
    node = pt.from_value(x)

    result = pt.map(lambda x: x * 2, node)
    assert result.fields[0].leaf_value == 2
    assert result.fields[1].leaf_value == 4
    assert result.fields[2].fields['a'].leaf_value == 6
    assert result.fields[2].fields['b'].leaf_value == 8

    # Test mapping multiple trees
    x2 = [10, 20, {'a': 30, 'b': 40}]
    node2 = pt.from_value(x2)

    result = pt.map(lambda x, y: x + y, node, node2)
    assert result.fields[0].leaf_value == 11
    assert result.fields[1].leaf_value == 22
    assert result.fields[2].fields['a'].leaf_value == 33
    assert result.fields[2].fields['b'].leaf_value == 44

    # Test error for mismatched tree structures
    y = {'c': 5}
    node3 = pt.from_value(y)
    try:
        result = pt.map(lambda x, y: x + y, node, node3)
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_pytree():
    # Test leaf nodes
    x = np.array([1, 2, 3])
    node = pt.leaf(x)
    assert node.leaf_value is x
    assert not node.fields
    assert node.typ is np.ndarray

    # Test sequence nodes
    lst = [1, 2, 3]
    node = pt.from_sequence(lst)
    assert node.typ is list
    assert len(node.fields) == 3
    assert all(isinstance(v, pt.Node) for v in node.fields.values())

    # Test dict nodes
    d = {'a': 1, 'b': 2}
    node = pt.from_dict(d)
    assert node.typ is dict
    assert set(node.fields.keys()) == {'a', 'b'}

    # Test nested structures
    nested = {'x': [1, 2], 'y': {'z': 3}}
    node = pt.from_value(nested)
    assert node.typ is dict
    assert set(node.fields.keys()) == {'x', 'y'}
    assert node.fields['x'].typ is list
    assert node.fields['y'].typ is dict

    # Test map function
    n1 = pt.leaf(1)
    n2 = pt.leaf(2)
    result = pt.map(lambda x,y: x+y, n1, n2)
    assert result.leaf_value == 3

def test_operators():
    # Test add operator
    @ad.jit
    def f_add(x, y):
        return x + y

    expr = f_add.get_expr(
        pt.leaf(ad.ArgSpec(np.float64, ())),
        pt.leaf(ad.ArgSpec(np.float64, ()))
    )
    interpreter = ad.Interpreter((pt.leaf(np.array(2.0)),
                                pt.leaf(np.array(3.0))))
    result = interpreter(expr)
    assert abs(result - 5.0) < 1e-6

    # Test subtract operator
    @ad.jit
    def f_sub(x, y):
        return x - y

    expr = f_sub.get_expr(
        pt.leaf(ad.ArgSpec(np.float64, ())),
        pt.leaf(ad.ArgSpec(np.float64, ()))
    )
    interpreter = ad.Interpreter((pt.leaf(np.array(5.0)),
                                pt.leaf(np.array(3.0))))
    result = interpreter(expr)
    assert abs(result - 2.0) < 1e-6

    # Test multiply operator
    @ad.jit
    def f_mul(x, y):
        return x * y

    expr = f_mul.get_expr(
        pt.leaf(ad.ArgSpec(np.float64, ())),
        pt.leaf(ad.ArgSpec(np.float64, ()))
    )
    interpreter = ad.Interpreter((pt.leaf(np.array(2.0)),
                                pt.leaf(np.array(3.0))))
    result = interpreter(expr)
    assert abs(result - 6.0) < 1e-6

    # Test power operator
    @ad.jit
    def f_pow(x):
        return x ** 2

    expr = f_pow.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    result = interpreter(expr)
    assert abs(result - 9.0) < 1e-6

    # Test negation operator
    @ad.jit
    def f_neg(x):
        return -x

    expr = f_neg.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    result = interpreter(expr)
    assert abs(result - (-3.0)) < 1e-6

    # Test exp operator
    @ad.jit
    def f_exp(x):
        return ad.exp(x)

    expr = f_exp.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(0.0)),))
    result = interpreter(expr)
    assert abs(result - 1.0) < 1e-6

    # Test sin operator
    @ad.jit
    def f_sin(x):
        return ad.sin(x)

    expr = f_sin.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(0.0)),))
    result = interpreter(expr)
    assert abs(result - 0.0) < 1e-6

    # Test cos operator
    @ad.jit
    def f_cos(x):
        return ad.cos(x)

    expr = f_cos.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(0.0)),))
    result = interpreter(expr)
    assert abs(result - 1.0) < 1e-6

    # Test abs operator
    @ad.jit
    def f_abs(x):
        return ad.abs(x)

    expr = f_abs.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(-3.0)),))
    result = interpreter(expr)
    assert abs(result - 3.0) < 1e-6

    # Test log operator
    @ad.jit
    def f_log(x):
        return ad.log(x)

    expr = f_log.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    interpreter = ad.Interpreter((pt.leaf(np.array(1.0)),))
    result = interpreter(expr)
    assert abs(result - 0.0) < 1e-6

    # Test matmul operator
    @ad.jit
    def f_matmul(x, y):
        return x @ y

    expr = f_matmul.get_expr(
        pt.leaf(ad.ArgSpec(np.float64, (2, 2))),
        pt.leaf(ad.ArgSpec(np.float64, (2, 1)))
    )
    interpreter = ad.Interpreter((
        pt.leaf(np.array([[1.0, 2.0], [3.0, 4.0]])),
        pt.leaf(np.array([[5.0], [6.0]]))
    ))
    result = interpreter(expr)
    assert np.allclose(result, np.array([[17.0], [39.0]]))

# Test autodiff.py
def test_gradient():
    # Test basic gradient computation
    @ad.jit
    def f(x):
        return x ** 2

    expr = f.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    grad_expr = ad.Gradient()(expr)

    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    grad_val = interpreter(grad_expr)
    expected = 2 * 3.0 # df/dx = 2x

    assert abs(grad_val - expected) < 1e-6

def test_multivariate():
    # Test multivariate gradient
    @ad.jit
    def f(x, y):
        return x * y + x ** 2

    expr = f.get_expr(
        pt.leaf(ad.ArgSpec(np.float64, ())),
        pt.leaf(ad.ArgSpec(np.float64, ()))
    )
    grad_expr = ad.Gradient()(expr)
    print(grad_expr)

    interpreter = ad.Interpreter((pt.leaf(np.array(2.0)),
                             pt.leaf(np.array(3.0))))
    dx, dy = interpreter(grad_expr)

    assert abs(dx - (3 + 4)) < 1e-6  # df/dx = y + 2x = 3 + 4
    assert abs(dy - 2) < 1e-6        # df/dy = x = 2

def test_dead_code_elimination():
    @ad.jit
    def f(x):
        # Create dead code that shouldn't affect result
        _y = x * 2
        _z = _y + 1
        return [_z, x ** 2][1]

    expr = f.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    optimized = ad.DeadCodeElimination()(expr)

    # Test both give same result
    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    orig_result = interpreter(expr)
    opt_result = interpreter(optimized)
    assert abs(orig_result - opt_result) < 1e-6

def test_common_subexpression_elimination():
    @ad.jit
    def f(x):
        # Create common subexpressions
        y = x * 2
        z = x * 2  # Same computation as y
        return y + z

    expr = f.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    optimized = ad.CommonSubexpressionElimination()(expr)

    # Test both give same result
    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    orig_result = interpreter(expr)
    opt_result = interpreter(optimized)
    assert abs(orig_result - opt_result) < 1e-6

def test_constant_folding():
    @ad.jit
    def f(x):
        # Create constant expressions that can be folded
        y = 2.0 * 3.0  # Should fold to 6.0
        return x * y

    expr = f.get_expr(pt.leaf(ad.ArgSpec(np.float64, ())))
    optimized = ad.ConstantFolding()(expr)

    # Test both give same result
    interpreter = ad.Interpreter((pt.leaf(np.array(3.0)),))
    orig_result = interpreter(expr)
    opt_result = interpreter(optimized)
    assert abs(orig_result - opt_result) < 1e-6
