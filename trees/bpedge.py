from trees.edge import Edge

__author__ = 'sim'


class BPEdge(Edge):
    """ Specialized Edge for Belief propagation """

    def __init__(self, parent, child):
        super().__init__(parent, child)
        # "transition" probs
        self.potentials = None  #
        self.up_msg = None  #
        self.max_up_msg = None  # for max-product msg passing
        self.max_paths = None  # for max-product msg passing
        self.down_msg = None  #
        self.posterior = None

    def set_potentials(self, potentials):
        self.potentials = potentials

    def clear_edge(self):
        self.potentials = None  #
        self.up_msg = None  #
        self.max_up_msg = None  # for max-product msg passing
        self.max_paths = None  # for max-product msg passing
        self.down_msg = None  #
        self.posterior = None