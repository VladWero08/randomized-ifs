class ExternalNode:
    def __init__(self, size: int, data):
        self.size = size 
        self.data = data

class InternalNode:
    pass

class InternalNode:
    def __init__(
        self, 
        left: ExternalNode | InternalNode, 
        right: ExternalNode | InternalNode,
        split_attribute: int,
        split_value: int | float
    ):
        self.left = left
        self.right = right
        self.split_attribute = split_attribute
        self.split_value = split_value