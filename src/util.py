class Node:
    left = None
    right = None
    attr = None
    decision = None

    def __init__(self, attr, decision):
        self.attr = attr
        self.decision = decision


class Tree:
    root = None

    def __init__(self, root):
        self.root = root
