import hypothesis.strategies as st
from hypothesis import given
from dataclasses import dataclass
from typing import Any, List
import slimdiffy.pytree as pt

atomic_types = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.none(),
    st.booleans()
)

def make_pytree_strategy(max_depth=3):
    """
    Creates a strategy for generating PyTree-like objects with bounded depth.
    A PyTree can be:
    - A leaf value (int, float, str, etc.)
    - A list/tuple of PyTrees
    - A dict of str -> PyTree
    - A dataclass containing PyTrees
    """

    @dataclass
    class SimpleNode:
        value: Any
        children: List[Any]

    # Define our recursive strategy
    return st.recursive(
        # Base case - our leaf values
        atomic_types,

        # Recursive case - wrap in containers
        lambda children: st.one_of(
            # Lists of trees
            st.lists(children, max_size=4),

            # Tuples of trees (fixed and variable length)
            st.tuples(children, children),
            st.lists(children, max_size=4).map(tuple),

            # Dictionaries of trees
            st.dictionaries(st.text(min_size=1), children, max_size=4),

            # Dataclass containing trees
            st.builds(
                SimpleNode,
                children,
                st.lists(children, max_size=2)
            ),

            # Optional trees
            st.none() | children,
        ),
        max_leaves=max_depth
    )

make_node_strategy = lambda x: make_pytree_strategy(x).map(pt.from_value)

@given(make_pytree_strategy())
def test_to_from_value(tree):
    assert pt.from_value(tree).to_value() == tree

@given(make_pytree_strategy())
def test_leaf_to_value(tree):
    # Note 'tree' is misleading here, we're
    # just using it as a source of things
    assert pt.leaf(tree).to_value() == tree

@given(st.dictionaries(st.text(min_size=1), make_pytree_strategy(2)))
def test_from_dict_to_value(dict):
    assert pt.from_dict(dict).to_value() == dict

@given(st.dictionaries(st.text(min_size=1), make_node_strategy(2)))
def test_from_dict_to_value2(dict):
    expected = {k: v.to_value() for k, v in dict.items()}
    assert pt.from_dict(dict).to_value() == expected

@given(st.lists(make_pytree_strategy(2)), st.lists(st.booleans()))
def test_from_sequence_to_value(lst, mask):
    # Extend mask if needed
    mask = mask + [False] * (len(lst) - len(mask))
    assert pt.from_sequence(lst, lambda i, _: mask[i]).to_value() == lst

@given(st.lists(make_node_strategy(2)), st.lists(st.booleans()))
def test_from_sequence_to_value2(lst, mask):
    # Extend mask if needed
    mask = mask + [False] * (len(lst) - len(mask))
    expected = [v if not mask[i] else v.to_value() for i, v in enumerate(lst)]
    assert pt.from_sequence(lst, lambda i, _: mask[i]).to_value() == expected

@given(make_node_strategy(3))
def test_map_identity(tree):
    node = pt.from_value(tree)
    mapped = pt.map(lambda x: x, node)
    assert mapped.to_value() == tree

@given(make_node_strategy(3))
def test_map_composition(tree):
    # Map leaves to unique numbers
    count = 0
    def get_unique():
        nonlocal count
        count += 1
        return count

    node = pt.map(lambda _: get_unique(), tree)

    # Define functions that multiply by different primes
    def f(x): return x * 2
    def g(x): return x * 3

    # Test composition is same as separate maps
    composed = pt.map(lambda x: f(g(x)), node)
    separate = pt.map(f, pt.map(g, node))

    assert composed.to_value() == separate.to_value()

@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_map_dict_keys(dict):
    visited_keys = []
    def track_keys(x):
        visited_keys.append(x)
        return x
    node = pt.from_value(dict)
    pt.mapkeys(lambda path, _: track_keys(path[-1]), node)
    assert len(visited_keys) == len(dict.keys())
    assert set(visited_keys) == set(dict.keys())

@given(st.dictionaries(st.text(min_size=1), st.dictionaries(st.text(min_size=1), st.integers())))
def test_map_nested_dict_keys(dict):
    visited_keys = []
    def track_keys(path, _):
        visited_keys.append('.'.join(str(x) for x in path))
        return _
    node = pt.from_value(dict)
    pt.mapkeys(track_keys, node)
    expected_keys = []
    for k1, v1 in dict.items():
        for k2 in v1.keys():
            expected_keys.append(f"{k1}.{k2}")
    assert len(visited_keys) == len(expected_keys)
    assert set(visited_keys) == set(expected_keys)

@given(st.lists(st.integers()))
def test_map_sequence_keys(seq):
    visited_keys = []
    def track_keys(x):
        visited_keys.append(x)
        return x
    node = pt.from_value(seq)
    pt.mapkeys(lambda path, _: track_keys(path[-1]), node)
    assert visited_keys == list(range(len(seq)))

@given(st.lists(make_node_strategy(2)))
def test_sequence_roundtrip(lst):
    # Create a sequence of nodes and convert to value
    node = pt.from_sequence(lst)
    sequence = node.to_sequence()
    assert sequence == lst

def traverse_frozen_tree(tree, frozen_tree):
    if isinstance(tree, (int, float, str, bool, type(None), type)):
        return tree == frozen_tree
    elif isinstance(tree, (list, tuple)):
        if not isinstance(frozen_tree, tuple):
            return False
        if len(tree) != len(frozen_tree):
            return False
        return all(traverse_frozen_tree(t, f) for t, f in zip(tree, frozen_tree))
    elif isinstance(tree, dict):
        if not isinstance(frozen_tree, tuple):
            return False
        # Convert frozen dict form (tuple of (k,v) pairs) back to dict
        frozen_dict = dict(frozen_tree)
        if tree.keys() != frozen_dict.keys():
            return False
        return all(traverse_frozen_tree(tree[k], frozen_dict[k]) for k in tree)
    else:
        return True # Skip checking non-basic types

@given(make_pytree_strategy())
def test_freeze_tree(tree):
    frozen = pt.freeze(tree)

    # Test hashability
    try:
        hash(frozen)
    except TypeError:
        assert False, "Frozen tree should be hashable"

    # Test equality comparison
    frozen2 = pt.freeze(tree)
    assert frozen == frozen2

    # Test structure matches original
    assert traverse_frozen_tree(tree, frozen)
