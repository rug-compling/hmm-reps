from trees.node import Node

__author__ = 'sim'


class BPNode(Node):
    """ Specialized Node for Belief propagation """

    def __init__(self, index, name, rel=None):
        super().__init__(index, name, rel=rel)
        # "emission" probs, local evidence
        self.potentials = None
        # starting potential (special case of edge potential)
        self.initial_potentials = None  # has value only for leaf
        self.up_belief = None
        self.max_up_belief = None  # for max-product msg passing
        self.max_state = None  # for max-product msg passing
        self.down_belief = None
        self.posterior = None

    def set_initial_potentials(self, potentials):
        self.initial_potentials = potentials

    def set_potentials(self, potentials):
        self.potentials = potentials

    def is_ready(self, tree):
        """
        Node is ready when messages from all its children are available
        """
        # get all edges from children
        # edges = [Tree.get_edge_by_nodes(self, node) for node in self.get_children()]
        #for edge in edges:
        #    if edge.up_msg is None:
        #        return False
        #return True
        for child in self.get_children():
            if tree.get_edge_by_nodes(self, child).up_msg is None:
                return False
        return True

    def is_ready_decoding(self, tree):
        """
        Node is ready when messages from all its children from max-product decoding are available
        """
        # get all edges from children
        # edges = [Tree.get_edge_by_nodes(self, node) for node in self.get_children()]
        #for edge in edges:
        #    if edge.up_msg is None:
        #        return False
        #return True
        for child in self.get_children():
            if tree.get_edge_by_nodes(self, child).max_up_msg is None:
                return False
        return True

    def clear_node(self):
        self.potentials = None
        self.initial_potentials = None
        self.up_belief = None
        self.max_up_belief = None  # for max-product msg passing
        self.max_state = None
        self.down_belief = None
        self.posterior = None