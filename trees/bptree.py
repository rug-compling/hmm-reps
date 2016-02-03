from trees.tree import Tree

__author__ = 'sim'


class BPTree(Tree):
    def __init__(self):
        super().__init__()
        self.ll = None
        # self.max_state_list = []

    def set_ll(self, ll):
        self.ll = ll

    def get_ll(self):
        return self.ll

    def clear_ll(self):
        self.ll = None

    def clear_tree(self):
        """
        clear up all BP-related information
        """
        for node in self:
            node.clear_node()
        for edge in self.get_edges():
            edge.clear_edge()
        self.max_state_list = []
        self.clear_ll()

    def shallow_copy(self):
        t = BPTree()
        t.ll = self.ll
        t.node_list = self.node_list[:]
        t.edge_list = self.edge_list[:]
        t.nodes_to_edge = self.nodes_to_edge.copy()

        return t