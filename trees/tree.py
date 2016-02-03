from trees.edge import Edge
from trees.node import Node


class Tree:
    def __init__(self):
        self.node_list = []
        self.edge_list = []
        self.nodes_to_edge = {}

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        """
        Get node with index from tree
        node with idx 0 should be root
        """
        for node in self.node_list:
            if node.get_index() == idx:
                return node

    def __iter__(self):
        return iter(self.node_list)

    def __contains__(self, idx):
        for node in self.node_list:
            if node.get_index() == idx:
                return True
        return False

    def copy(self):
        # t = Tree()
        #t.node_list = self.node_list[:]
        #t.edge_list = self.edge_list[:]
        #t.nodes_to_edge = self.nodes_to_edge.copy()
        #return t
        raise NotImplementedError()

    def add_node(self, node):
        if isinstance(node, Node):
            self.node_list.append(node)
        else:
            raise TypeError

    def add_edge(self, edge):
        if isinstance(edge, Edge):
            self.edge_list.append(edge)
        else:
            raise TypeError

    def get_root(self):
        """
         assume 1 root
         """
        for node in self.node_list:
            if not node.has_parent():
                return node

    def get_leaves(self):
        return [node for node in self if not node.has_children()]

    def get_nonroots(self):
        """
         All nodes that are not root
        """
        root = self.get_root()
        return [node for node in self if node != root]

    def get_node(self, idx):
        return self.__getitem__(idx)

    def get_nodes(self):
        return self.node_list

    def get_relations(self):
        root = self.get_root()
        return [node.rel for node in self if node != root]

    def get_edges(self):
        return self.edge_list

    def get_edges_to_root(self):
        """
        All edges leading to root
        """
        root = self.get_root()
        return [edge for edge in self.edge_list if edge.get_parent() == root]

    def get_edges_not_to_root(self):
        """
        All edges not leading to root
        """
        root = self.get_root()
        return [edge for edge in self.edge_list if edge.get_parent() != root]

    def get_edges_where_parent(self, node):
        """
        All edges in which node is the parent.
        """
        return [edge for edge in self.get_edges() if edge.get_parent() == node]

    def get_num_edges(self):
        return len(self.edge_list)

    def add_edge_to_map(self, parent, child, edge):
        """
        update the nodes_to_edge map
        """
        self.nodes_to_edge[(parent, child)] = edge

    def get_edge_by_nodes_slow(self, node1, node2):
        # assert node1 != node2
        for edge in self.get_edges():
            if edge.get_parent_and_child() == {node1, node2}:
                return edge

    def get_edge_by_nodes(self, parent, child):
        return self.nodes_to_edge[(parent, child)]

    def print_tree_edges(self):
        """
        conll-like: child id with corresponding parent id
        """
        return [(e.get_child().get_index(), e.get_parent().get_index()) for e in self.get_edges()]

    def print_tree_edges_with_name(self):
        """
        conll-like: child id with corresponding parent id
        """
        return [(e.get_child().get_name(), e.get_parent().get_name()) for e in self.get_edges()]

    def is_connected(self):
        """
        catch corrupted, disconnected "tree" (forest)

        if multiple roots, it is disconnected
        """
        roots = []
        for node in self.node_list:
            if not node.has_parent():
                roots.append(node)
        if len(roots) > 1:
            print("Disconnected tree caught")
            return False
        else:
            return True

    def is_acyclic(self):
        """
        depth-first search from leaves to root, if same element never encountered twice, is acyclic
        approximate: would fail in case of a simple cycle without leaf
        """
        leaves = self.get_leaves()
        for l in leaves:
            seen = set()
            seen.add(l)
            par = l.get_parent()
            while par is not None:
                if par in seen:
                    return False
                seen.add(par)
                par = par.get_parent()
        return True