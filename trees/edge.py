__author__ = 'sim'


class Edge:
    def __init__(self, parent, child):
        """
        :param parent: Node object parent
        :param child: Node object child
        """
        self.parent = parent
        self.child = child

        # assert self.parent != self.child, "Child not allowed to be its own parent."

        #update nodes
        self.parent.add_child(self.child)
        self.child.set_parent(self.parent)

    def get_parent(self):
        return self.parent

    def get_child(self):
        return self.child

    def get_parent_and_child(self):
        return {self.parent, self.child}