import numpy as np

from inference.hmm_inference import project_kbest
from util.log_domain import logzero, sselogsum


__author__ = 'sim'


class SumProduct:
    """ Sum-product message passing algorithm for inference on trees.
    This implementation follows the exposition in [1] (section 20.2.1.)

    [1] Kevin P. Murphy (2012) Machine learning : a probabilistic perspective. (Ch. 20: Exact inference
    for graphical models)

    """

    def __init__(self, approximate=False):
        self.approximate = approximate

    def compute_posteriors(self, tree, N):
        """
        Compute state and transition posteriors.

        State posterior is the probability of each state at each
        node in the tree given all observed variables. Transition posterior
        is the joint probability of being in two nodes at the same time
        given all observed variables

        To compute the two posteriors, we propagate messages in the tree upward
        and backward using sum product aka belief propagation. The goal is to fill
        in the tree score structure (from treerepr_scores()). Concretely, we use the
        following:
         - upward belief at each node
         - upward message at each edge
         - downward message at each edge

        Node posterior is a product of upward belief and the downward message normalized
        by log likelihood.
        Edge posterior between a child and a parent is the product of:
         upward belief of the child,
         edge potentials between the child and the parent,
         the parent potentials,
         the downward message to the parent,
         the product of upward messages from all other children to the parent,
        normalized by log likelihood.
        When the parent is the root node, the calculation excludes the parent potentials and
        the downward message to the parent

        :param tree: tree object defined on BPnode and BPedge objects
        :param approximate: whether to use projected/approximated vectors as regularization and speed-up
        """

        # upward and downward propagation
        self.up_propagate(tree, N)
        tree.set_ll(tree.get_root().up_belief)
        ll = tree.get_ll()

        self.down_propagate(tree, N)

        for node in tree.get_nonroots():
            parent = node.get_parent()
            curr_edge = tree.get_edge_by_nodes(parent, node)
            node.posterior = self.compute_node_posterior(node, curr_edge, ll)
            curr_edge.posterior = self.compute_edge_posterior(node, parent, curr_edge, tree, ll)

    def up_propagate(self, tree, N):
        """
        compute upward belief at each node (function of incoming msgs and node potential) and
        send the message to the parent
        """
        root = tree.get_root()
        # assert len(root.get_children()) == 1  # not necessary
        active_nodes = tree.get_leaves()
        while active_nodes:
            curr_node = active_nodes.pop()
            #compute belief if it doesn't exist
            if curr_node.up_belief is None:
                curr_node.up_belief = self.compute_belief(curr_node, tree)
                #print("node {} upbelief:\n{}".format(curr_node.get_name(), curr_node.up_belief))
            if curr_node != root:
                self.pass_msg_up(tree, curr_node, curr_node.get_parent(), N)
                if curr_node.get_parent().is_ready(tree):
                    active_nodes.append(curr_node.get_parent())

    def down_propagate(self, tree, N):
        """
        compute and pass downward messages to every child
        """
        root = tree.get_root()
        # assert len(root.get_children()) == 1
        active_nodes = [root]
        while active_nodes:
            curr_node = active_nodes.pop()
            if not curr_node.has_children():
                continue
            else:
                for child in curr_node.get_children():
                    self.pass_msg_down(curr_node, tree, N, child)
                    active_nodes.append(child)

    def compute_belief(self, node, tree):
        """
        belief is the product of
         node's potential (emission prob) and
         the product of messages sent to the node
        """
        if not node.has_children():  # is leaf
            product = node.initial_potentials
        else:
            product = sum([tree.get_edge_by_nodes(node, child).up_msg for child in node.get_children()])
        if node.is_root():  # no potential for root
            return product
        else:
            return project_kbest(product + node.potentials) if self.approximate else product + node.potentials

    def pass_msg_up(self, tree, sender, receiver, N):
        """
        upward message is the sum over all states of the product of
         the upward belief of the sender and
         the sender-receiver edge potential

        :param sender: child
        :param receiver: parent
        """
        # edge to store the message
        curr_edge = tree.get_edge_by_nodes(receiver, sender)
        up_msg_temp = np.zeros(N, 'f') + logzero()

        if receiver.is_root():
            # edge potential here only Nx1
            up_msg_temp = sselogsum(sender.up_belief + curr_edge.potentials)
        else:
            for curr_state in range(N):  # vectorize!
                # print("curr_edge.potentials[{}, :]\n{}".format(curr_state, curr_edge.potentials[curr_state, :]))
                up_msg_temp[curr_state] = sselogsum(sender.up_belief + curr_edge.potentials[curr_state, :])

        # curr_edge.up_msg = sparse.csr_matrix(up_msg_temp)
        curr_edge.up_msg = up_msg_temp

    def pass_msg_down(self, sender, tree, N, receiver):
        """
        downward message is the sum over all states of the product of
         the downward message from sender's parent,
         the product of upward messages to the sender from his other children,
         the sender-receiver edge potential,
         the sender node potential
        If the sender is the root, the calculation is somewhat simplified.

        :param sender: parent
        :param receiver: child
        """
        down_msg_temp = np.zeros(N, 'f') + logzero()
        curr_edge = tree.get_edge_by_nodes(sender, receiver)
        # can be zero if there's not children
        product_child = sum([tree.get_edge_by_nodes(sender, c).up_msg
                             for c in sender.get_children()
                             if c != receiver])

        if sender.is_root():
            down_msg_temp = curr_edge.potentials + product_child
        else:
            prev_edge = tree.get_edge_by_nodes(sender.get_parent(), sender)
            product = project_kbest(
                prev_edge.down_msg + product_child) if self.approximate else prev_edge.down_msg + product_child
            product += sender.potentials
            for curr_state in range(N):  # TODO optimize
                down_msg_temp[curr_state] = sselogsum(product + curr_edge.potentials[:, curr_state])

        curr_edge.down_msg = down_msg_temp

    def run_max_product(self, tree, N):
        """
        Performs max-product (i.e. hard belief propagation/tree Viterbi).
        In log-space: max-sum.

        See Bishop p.412-415.

        :param N: number of states
        """
        # initialize max_up_belief (will replace up_belief in computation)

        # backtracking?

        # most likely state for each nonroot node

        # ###### up_propagate:
        # """
        #compute upward belief at each node (function of incoming msgs and node potential) and
        #send the message to the parent
        #"""
        root = tree.get_root()

        active_nodes = tree.get_leaves()

        while active_nodes:
            curr_node = active_nodes.pop()
            #compute max belief if it doesn't exist
            if curr_node.max_up_belief is None:
                curr_node.max_up_belief = self.compute_max_belief(curr_node, tree)
            if curr_node != root:
                self.pass_max_msg_up(tree, curr_node, curr_node.get_parent(), N)
                if curr_node.get_parent().is_ready_decoding(tree):
                    active_nodes.append(curr_node.get_parent())

        # Backtrack
        max_states = {}
        active_edges = tree.get_edges_to_root()
        while active_edges:
            curr_edge = active_edges.pop()
            curr_child = curr_edge.get_child()
            if curr_edge in tree.get_edges_to_root():
                curr_child.max_state = curr_edge.max_paths  # scalar
                max_states[curr_child.index] = curr_child.max_state
            else:
                curr_child.max_state = curr_edge.max_paths[curr_edge.get_parent().max_state]
                max_states[curr_child.index] = curr_child.max_state
            active_edges.extend(tree.get_edges_where_parent(curr_child))

        return max_states

    def run_max_posterior(self, tree, N, cont, ignore_rel=None):
        """
        Compute maximum-scoring states from state posteriors.

        State posterior is the probability of each state at each
        node in the tree given all observed variables.

        :param tree: tree object defined on BPnode and BPedge objects
        :param cont: whether to return continuous (distribution) rep or discrete state
        :param ignore_rel: do not output state if dep. relation is same as ignore_rel
        """
        # upward and downward propagation
        self.up_propagate(tree, N)
        tree.set_ll(tree.get_root().up_belief)
        ll = tree.get_ll()

        self.down_propagate(tree, N)
        best_states = {}
        # get posteriors
        for node in tree.get_nonroots():
            if node.rel == ignore_rel:
                continue
            parent = node.get_parent()
            curr_edge = tree.get_edge_by_nodes(parent, node)
            node.posterior = self.compute_node_posterior(node, curr_edge, ll)
            # assert len(node.posterior.shape) == 1
            if cont:
                node.max_state = node.posterior
            else:
                node.max_state = node.posterior.argmax()
            best_states[node.index] = node.max_state

        return best_states

    def compute_node_posterior(self, node, curr_edge, ll):
        node.posterior = (node.up_belief + curr_edge.down_msg) - ll
        return np.exp(node.posterior)

    def compute_edge_posterior(self, node, parent, curr_edge, tree, ll):
        # edge posterior¸
        if parent.is_root():
            product_child = [tree.get_edge_by_nodes(parent, c).up_msg
                             for c in parent.get_children()
                             if c != node]
            if product_child:
                product_child = sum(product_child)
            curr_edge.posterior = (node.up_belief + curr_edge.potentials)
            if product_child != []:
                curr_edge.posterior += product_child
        else:
            msg_parent = tree.get_edge_by_nodes(parent.get_parent(), parent).down_msg

            product_child = [tree.get_edge_by_nodes(parent, c).up_msg
                             for c in parent.get_children()
                             if c != node]
            if product_child:
                product_child = sum(product_child)
            # print("upbelief:\n{}\ncurr_edge-potent:\n{}\nparent potential:\n{}\nproduct child:\n{}\nmsg_parent\n{}".format(node.up_belief, curr_edge.potentials, parent.potentials, product_child, msg_parent))
            curr_edge.posterior = (node.up_belief +
                                   curr_edge.potentials +
                                   parent.potentials.reshape(-1, 1) +
                                   msg_parent.reshape(-1, 1))
            if product_child != []:
                curr_edge.posterior += product_child.reshape(-1, 1)

        curr_edge.posterior -= ll
        return np.exp(curr_edge.posterior)

    def compute_max_belief(self, node, tree):
        """
        For max-product.

        belief is the product of
         node's potential (emission prob) and
         the product of messages sent to the node
        """
        if not node.has_children():  # is leaf
            product = node.initial_potentials
        else:
            product = sum([tree.get_edge_by_nodes(node, child).max_up_msg for child in node.get_children()])
        if node.is_root():  # no potential for root
            # if len(node.get_children()) == 2:
            #    print([tree.get_edge_by_nodes(node, child).up_msg for child in node.get_children()])
            return product
        else:
            return product + node.potentials

    def pass_max_msg_up(self, tree, sender, receiver, N):
        """
        For max-product.

        upward message is the sum of product of
         the upward belief of the sender and
         the sender-receiver edge potential

        :param sender: child
        :param receiver: parent
        """
        # edge to store the message
        curr_edge = tree.get_edge_by_nodes(receiver, sender)
        max_up_msg_temp = np.zeros(N, 'f') + logzero()
        max_paths_temp = -np.ones(N, dtype=int)

        if receiver.is_root():
            # edge potential here only Nx1
            max_up_msg_temp = np.max(sender.max_up_belief + curr_edge.potentials)
            max_paths_temp = np.argmax(sender.max_up_belief + curr_edge.potentials)
        else:
            for curr_state in range(N):  # vectorize!
                # print("curr_edge.potentials[{}, :]\n{}".format(curr_state, curr_edge.potentials[curr_state, :]))
                max_up_msg_temp[curr_state] = np.max(sender.max_up_belief + curr_edge.potentials[curr_state, :])
                max_paths_temp[curr_state] = np.argmax(sender.max_up_belief + curr_edge.potentials[curr_state, :])

        # NOTE TODO;
        #### up_msg could keep argmaxes

        curr_edge.max_up_msg = max_up_msg_temp
        curr_edge.max_paths = max_paths_temp