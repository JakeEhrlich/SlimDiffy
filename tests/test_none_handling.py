import slimdiffy.pytree as pt

def test_none_leaf():
    """Test None as a leaf value"""
    print('Testing None as leaf value:')
    try:
        node = pt.leaf(None)
        value = node.to_value()
        print(f'Success: None leaf value converted back to {value}')
    except Exception as e:
        print(f'Error: {e}')

def test_none_in_sequence():
    """Test None in a sequence"""
    print('\nTesting None in sequence:')
    try:
        node = pt.from_value([1, None, 3])
        value = node.to_value()
        print(f'Success: List with None converted back to {value}')
    except Exception as e:
        print(f'Error: {e}')

def test_none_in_dict():
    """Test None in a dictionary"""
    print('\nTesting None in dictionary:')
    try:
        node = pt.from_value({'a': None, 'b': 2})
        value = node.to_value()
        print(f'Success: Dict with None converted back to {value}')
    except Exception as e:
        print(f'Error: {e}')

def test_none_type_node():
    """Test Node with NoneType as typ"""
    try:
        node = pt.Node(type(None), {}, {})
        value = node.to_value()
        print(f'Success: Node with NoneType typ converted to {value}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_none_leaf()
    test_none_in_sequence()
    test_none_in_dict()
    test_none_type_node()
