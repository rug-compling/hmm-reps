from sequences.label_dictionary import LabelDictionary
from trees.edge import Edge
from trees.node import Node
from trees.tree import Tree
from trees.tree_list import TreeList

__author__ = 'sim'


class ExampleTree:
    def __init__(self):
        self.x_dict = LabelDictionary(["write", "that", "code", "ROOT", "don't"])
        self.train_trees = TreeList()
        tree_ex1 = Tree()  # container for node_list and edge_list
        idx = self.x_dict.get_label_id("write")
        n0 = Node(len(tree_ex1), idx)  # len is 0
        tree_ex1.add_node(n0)
        idx = self.x_dict.get_label_id("that")
        n1 = Node(len(tree_ex1), idx)
        tree_ex1.add_node(n1)
        idx = self.x_dict.get_label_id("code")
        n2 = Node(len(tree_ex1), idx)
        tree_ex1.add_node(n2)
        idx = self.x_dict.get_label_id("ROOT")
        n3 = Node(len(tree_ex1), idx)
        tree_ex1.add_node(n3)

        tree_ex1.add_edge(Edge(n0, n2))
        tree_ex1.add_edge(Edge(n2, n1))
        tree_ex1.add_edge(Edge(n3, n0))

        self.train_trees.add_tree(tree_ex1)

        tree_ex2 = Tree()
        idx = self.x_dict.get_label_id("don't")
        n0 = Node(len(tree_ex1), idx)  # len is 0
        tree_ex2.add_node(n0)
        idx = self.x_dict.get_label_id("write")
        n1 = Node(len(tree_ex1), idx)
        tree_ex2.add_node(n1)
        idx = self.x_dict.get_label_id("code")
        n2 = Node(len(tree_ex1), idx)
        tree_ex2.add_node(n2)
        idx = self.x_dict.get_label_id("ROOT")
        n3 = Node(len(tree_ex1), idx)
        tree_ex2.add_node(n3)

        tree_ex2.add_edge(Edge(n0, n1))
        tree_ex2.add_edge(Edge(n1, n2))
        tree_ex2.add_edge(Edge(n3, n0))

        self.train_trees.add_tree(tree_ex2)