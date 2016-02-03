class TreeList:
    def __init__(self):
        self.tree_list = []

    def __len__(self):
        return len(self.tree_list)

    def __getitem__(self, item):
        return self.tree_list[item]

    def get_num_nodes(self):
        return sum([len(tree) for tree in self.tree_list])

    def get_num_edges(self):
        return sum([tree.get_num_edges() for tree in self.tree_list])

    def get_num_tokens(self):
        return self.get_num_nodes() - 1  # -root

    def add_tree(self, tree):
        self.tree_list.append(tree)