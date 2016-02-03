__author__ = 'sim'


class Node:
    def __init__(self, index, name, rel=None):
        """
        :param index: index of the node (makes sense for trees)
        :param name: name of the node, such as word or word index
        """
        self.index = index
        self.name = name
        self.rel = rel
        self.children = []
        self.parent = None

    def set_parent(self, parent):
        if isinstance(parent, Node):
            self.parent = parent
        else:
            raise TypeError

    def add_child(self, child):
        if isinstance(child, Node):
            self.children.append(child)
        else:
            raise TypeError

    def has_parent(self):
        return (self.parent is not None) or False

    def get_parent(self):
        return self.parent

    def is_root(self):
        return True if self.parent is None else False

    def has_children(self):
        return (bool(self.children)) or False

    def get_children(self):
        return self.children

    def get_index(self):
        return self.index

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_furthest_parent(self):
        """
        find ultimate parent

        for when there are fake roots in corrupted trees
        """
        raise NotImplementedError()