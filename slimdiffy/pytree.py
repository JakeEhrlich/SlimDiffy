# Copyright (c) 2024 Jake Ehrlich
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union

def static_field(value: Any) -> Any:
    return dataclasses.field(default=value, metadata={'static_field': True})

def leaf(value: Any) -> 'Node':
    """Creates a Node leaf containing the given value"""
    return Node(type(value), {}, {}, value)

def from_dict(d: Dict[str, Any]) -> 'Node':
    """Creates a Node from a dictionary of values"""
    return Node(dict, {k: leaf(v) if not isinstance(v, Node) else v
                            for k,v in d.items()}, {})

def from_sequence(s: Union[List, Tuple]) -> 'Node':
    """Creates a Node from a list or tuple"""
    return Node(type(s), {i: leaf(v) if not isinstance(v, Node) else v
                               for i,v in enumerate(s)}, {})

def mapkeys(f: Callable[..., Any], *trees: 'Node', path: Tuple[Union[str, int], ...] = ()) -> 'Node':
    """Maps a function over multiple Nodes, applying f to corresponding leaves"""
    leaf_values = [tree.leaf_value is not None for tree in trees]
    if any(leaf_values) and not all(leaf_values):
        raise ValueError("Trees must have leaves in same positions")

    if all(leaf_values):
        return leaf(f(path, *[tree.leaf_value for tree in trees]))

    # Check all trees have same structure
    if not all(tree.typ == trees[0].typ for tree in trees):
        raise ValueError("All trees must have same type")
    if not all(tree.fields.keys() == trees[0].fields.keys() for tree in trees):
        raise ValueError("All trees must have same field structure")

    # Map over fields recursively
    return Node(
        trees[0].typ,
        {k: mapkeys(f, *[tree.fields[k] for tree in trees], path=(*path, k))
         for k in trees[0].fields},
        trees[0].metadata
    )

def map(f: Callable[..., Any], *trees: 'Node') -> 'Node':
    """Maps a function over multiple Nodes, applying f to corresponding leaves"""
    def ignore_keys(_keys: Tuple[Union[str, int], ...], *args: Any) -> Any:
        return f(*args)
    return mapkeys(ignore_keys, *trees)

@dataclass
class Node:
    """Represents a node in a pytree structure"""
    typ: type
    fields: dict[Union[str, int], 'Node']
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    leaf_value: Any = None

    def __post_init__(self):
        if bool(self.fields) and self.leaf_value is not None:
            raise ValueError("Nodes cannot have both fields and a leaf value")

    def to_sequence(self) -> Union[List, Tuple]:
        """Converts a Nodes for a sequence into a sequence of Nodes"""
        if self.leaf_value is not None:
            return self.leaf_value
        result: list[Any] = [None] * len(self.fields)
        for i, v in self.fields.items():
            assert isinstance(i, int), f"Sequence index must be integer, got {type(i)}"
            result[i] = v
        assert None not in result, "Sequence cannot contain None values"
        return tuple(result) if self.typ is tuple else result

    def render(self, indent: int = 0, render_leaf: Callable = repr) -> str:
        """Renders the Nodes as a string with the specified indentation level"""
        if self.leaf_value is not None:
            return render_leaf(self.leaf_value)

        indent_str = " " * indent
        inner_indent = indent + 2

        if self.typ is list:
            items = ["\n" + " " * inner_indent + self.fields[i].render(inner_indent, render_leaf)
                    for i in range(len(self.fields))]
            return "[" + ",".join(items) + "\n" + indent_str + "]"

        elif self.typ is tuple:
            items = ["\n" + " " * inner_indent + self.fields[i].render(inner_indent, render_leaf)
                    for i in range(len(self.fields))]
            return "(" + ",".join(items) + "\n" + indent_str + ")"

        elif self.typ is dict:
            items = ["\n" + " " * inner_indent + repr(k) + ": " + v.render(inner_indent, render_leaf)
                    for k, v in self.fields.items()]
            return "{" + ",".join(items) + "\n" + indent_str + "}"

        else:
            items = ["\n" + " " * inner_indent + f"{k}={v.render(inner_indent, render_leaf)}"
                    for k, v in self.fields.items()]
            return f"{self.typ.__name__}({','.join(items)}\n{indent_str})"

    def to_value(self) -> Any:
        """Converts a Nodes back into the original value it represents"""
        if self.leaf_value is not None:
            return self.leaf_value

        if self.typ in (list, tuple):
            result: list[Any] = [None] * len(self.fields)
            for i, v in self.fields.items():
                assert type(i) is int
                result[i] = v.to_value()
            return tuple(result) if self.typ is tuple else result

        elif self.typ is dict:
            return {k: v.to_value() for k,v in self.fields.items()}

        elif dataclasses.is_dataclass(self.typ):
            args = {k: v.to_value() for k,v in self.fields.items()}
            args.update(self.metadata) # type: ignore
            return self.typ(**args) # type: ignore

        else:
            raise ValueError(f"Cannot convert Node with type {self.typ} to value")

def from_value(x: Any) -> Node:
    """Gets the Node structure of a value"""
    if isinstance(x, (list, tuple)):
        node = from_sequence([from_value(v) for v in x])
        if isinstance(x, tuple):
            node.typ = tuple
        return node
    elif isinstance(x, dict):
        return from_dict({k: from_value(v) for k,v in x.items()})
    elif dataclasses.is_dataclass(x):
        fields = {}
        metadata = {}
        for f in dataclasses.fields(x):
            if f.metadata.get('static_field', False):
                metadata[f.name] = getattr(x, f.name)
            else:
                fields[f.name] = from_value(getattr(x, f.name))
        return Node(type(x), fields, metadata) # type: ignore
    else:
        return leaf(x)

def freeze(x: Any) -> Any:
    """Converts a value into an immutable form suitable for dictionary keys"""
    if isinstance(x, (str, int, float, bool, complex, bytes, type(None))):
        return x
    elif isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    elif isinstance(x, dict):
        return tuple(sorted((freeze(k), freeze(v)) for k,v in x.items()))
    elif dataclasses.is_dataclass(x):
        fields = {}
        for f in dataclasses.fields(x):
            if not f.metadata.get('static_field', False):
                fields[f.name] = freeze(getattr(x, f.name))
        return tuple(sorted((k, v) for k,v in fields.items()))
    else:
        raise ValueError(f"Cannot freeze value of type {type(x)}")
