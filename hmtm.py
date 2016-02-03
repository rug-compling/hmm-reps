import os
import sys

import numpy as np

from hmm import HMM
from inference.sum_product import SumProduct
from util.util import chunk_seqs_by_size_gen


__author__ = 'sim'


class HMTM(HMM):
    """
    Hidden Markov Tree Model (for word clustering): EM variants with sum product message passing for inference
    """

    def __init__(self, N, M, params=None, writeout=False, brown_init_path=None, x_dict=None, approx=False,
                 dirname=None):
        super().__init__(N, M, params=params, writeout=writeout, brown_init_path=brown_init_path, x_dict=x_dict,
                         dirname=dirname)
        # sum-product instead of forward-backward
        self.inference = SumProduct(approximate=approx)

    def em_core(self, dataset, max_iter, logger):
        for it in range(1, max_iter + 1):
            # E-step
            total_ll = 0.0
            self.clear_counts()
            dataset.train = dataset.prepare_trees_gen()
            for c, tree in enumerate(dataset.train, 1):
                # prepare tree
                self.treerepr_scores(tree)
                # obtain node and edge posteriors and ll:
                self.inference.compute_posteriors(tree, self.N)
                #update counts for that tree
                self.update_counts_from_tree(tree)
                total_ll += tree.get_ll()
                tree.clear_tree()
                if c % 1000 == 0:
                    logger.info("{} sents".format(c))

            logger.info("Iter: {}\tLog-likelihood: {}".format(it, total_ll))
            self.lls.append(total_ll)

            # M-step
            self.compute_parameters(logger)

    def em_core_online(self, dataset, max_iter, logger):
        for it in range(1, max_iter + 1):
            total_ll = 0.0
            dataset.train = dataset.prepare_trees_gen()
            for t, minibatch in enumerate(chunk_seqs_by_size_gen(dataset.train, self.minibatch_size), 0):
                # E-step
                self.clear_counts()
                for tree in minibatch:
                    # prepare tree
                    self.treerepr_scores(tree)
                    #obtain node and edge posteriors and ll:
                    self.inference.compute_posteriors(tree, self.N)
                    #update counts for that tree
                    self.update_counts_from_tree(tree)
                    total_ll += tree.get_ll()
                    tree = None
                # M-step
                self.compute_online_parameters(t)
                logger.info("minibatch {} (size {})".format(t, self.minibatch_size))

            logger.info("Iter: {}\tLog-likelihood: {}".format(it, total_ll))
            self.lls.append(total_ll)

    def em_process_multiseq(self, trees):
        """
        Makes a local copy of count matrices, the worker updates them for all trees
        and finally returns them as yet another partial counts.
        """
        try:
            total_ll = 0
            initial_counts = self.initial_counts
            transition_counts = self.transition_counts
            final_counts = self.final_counts
            emission_counts = self.emission_counts

            c = 0
            for c, tree in enumerate(trees, 1):
                # prepare tree representation
                self.treerepr_scores(tree)
                # obtain node and edge posteriors and ll:
                self.inference.compute_posteriors(tree, self.N)

                # update temporary counts
                initial_counts += sum([leaf.posterior for leaf in tree.get_leaves()])
                for node in tree.get_nonroots():
                    emission_counts[node.get_name(), :] += node.posterior
                for edge in tree.get_edges_not_to_root():
                    transition_counts += edge.posterior
                final_counts += sum([edge.posterior for edge in tree.get_edges_to_root()])
                total_ll += tree.get_ll()

                tree.clear_tree()

            return initial_counts, transition_counts, final_counts, emission_counts, total_ll
        except KeyboardInterrupt:
            pass

    def trellis_scores(self, seq):
        raise NotImplementedError()

    def treerepr_scores(self, tree):
        """
        Tree-analogue to trellis_scores; potentials depend on the relation

        :param tree: tree graph

        """
        # every leaf gets initial_probs
        for leaf in tree.get_leaves():
            leaf.set_initial_potentials(np.log(self.initial_probs))
        # every node except root gets emission_probs
        for node in tree.get_nonroots():
            node.set_potentials(np.log(self.emission_probs[node.get_name(), :]))
        #every edge gets transition_probs
        for edge in tree.get_edges_not_to_root():
            edge.set_potentials(np.log(self.transition_probs))
        #every edge to # root gets final_probs
        for edge in tree.get_edges_to_root():
            edge.set_potentials(np.log(self.final_probs))

    def update_counts_from_tree(self, tree):
        """
        In E-step:
        Update the count matrices with partials from one tree

        BUG: can overflow because of the large log posteriors in the case of a huge tree
        get extremely big when taking exp
        TODO: fix by postponing the exp from compute_posteriors() until compute_parameters()
        """

        self.initial_counts += sum([leaf.posterior for leaf in tree.get_leaves()])
        for node in tree.get_nonroots():
            self.emission_counts[node.get_name(), :] += node.posterior
        for edge in tree.get_edges_not_to_root():
            self.transition_counts += edge.posterior

        self.final_counts += sum([edge.posterior for edge in tree.get_edges_to_root()])

    def max_product_decode(self, tree):
        """
        :param seq: classification sequence instance: tree available through seq.t
        """
        # prepare tree for inference
        self.treerepr_scores(tree)
        # obtain most likely states and ll:
        best_states = self.inference.run_max_product(tree, self.N)

        return best_states

    def max_product_decode_corpus(self, dataset):  # TODO: ignore relation option
        """Run Max-product (here, in log space: Max-sum) algorithm for decoding at corpus level.

        :param dataset: classification dataset with sequences, each containing tree as seq.t
        """

        for c, seq in enumerate(dataset.seq_list):
            if seq.t is None:
                seq.u = None
                continue
            # decode and store states to seq.t (tree)
            seq.u = self.max_product_decode(seq.t)
            seq.t = None

    def posterior_decode_corpus(self, dataset, ignore_rel=None):
        """
        :param dataset: classification dataset with sequences, each containing tree as seq.t
        """
        for c, seq in enumerate(dataset.seq_list):
            if seq.t is None:
                seq.u = None
                continue

            # decode and store states to seq.t (tree)
            seq.u = self.posterior_decode(seq.t, cont=False, ignore_rel=ignore_rel)
            seq.t = None

    def posterior_cont_decode_corpus(self, dataset, ignore_rel=None):
        """
        :param dataset: classification dataset with sequences, each containing tree as seq.t
        """
        for c, seq in enumerate(dataset.seq_list):
            if seq.t is None:
                seq.u = None
                continue

            # decode and store states to seq.t (tree)
            seq.u = self.posterior_decode(seq.t, cont=True, ignore_rel=ignore_rel)
            seq.t = None

    def posterior_decode(self, tree, cont, ignore_rel=None):
        """Compute the sequence of states that are individually the most
        probable, given the observations. This is done by maximizing
        the state posteriors, which are computed with message passing
        algorithm on trees."""
        # Compute scores given the observation sequence.
        self.treerepr_scores(tree)

        best_states = self.inference.run_max_posterior(tree, self.N, cont, ignore_rel=ignore_rel)
        return best_states

    def posterior_cont_type_decode_corpus(self, dataset, rep_dataset, logger=None):
        """Run posterior_decode at corpus level,
        return continuous rep per type (avg. over posteriors in all
        instances).

        :param dataset: evaluation dataset with seq objects
        """
        if self.posttypes is None:
            if self.dirname is not None:
                assert len(dataset.wordrep_dict) == len(rep_dataset.x_dict)
                posttype_f = "{}posttype.npy".format(self.dirname)
                self.posttypes = np.load(posttype_f) if os.path.exists(posttype_f) else self.obtain_posttypes(
                    posttype_f, rep_dataset, len(dataset.wordrep_dict), logger=logger)
                assert self.posttypes.shape == (len(dataset.wordrep_dict), self.N)
            else:
                sys.exit("dirname not set properly")

        if logger is not None: logger.info("Decoding on eval datasets.")
        # assign posteriors to types in dataset
        for seq in dataset.seq_list:
            if seq.t is None:
                print("seq.t is None")
                seq.u = None
                continue
            seq.u = {node.index: self.posttypes[node.name] for node in seq.t.get_nonroots()}
            # seq.t not needed, taking space; breaks however the __str__ methods which rely on seq.t
            seq.t = None

    def obtain_posttypes(self, posttype_f, rep_dataset, n_types, logger=None):
        if logger is not None: logger.info("Obtaining posterior type counts.")
        # obtain type posteriors
        type_posteriors = np.zeros((n_types, self.N))
        type_freq = np.zeros(n_types)
        for count, tree in enumerate(rep_dataset.train):
            # posteriors is dict with keys starting at 1
            if tree is None:
                continue

            if count % 1000 == 0:
                if logger is not None:
                    logger.debug(count)
                else:
                    print(count)

            posteriors = self.posterior_decode(tree, cont=True)
            for node in tree.get_nonroots():
                type_posteriors[node.name, :] += posteriors[node.index]
                type_freq[node.name] += 1
        #normalize
        type_posteriors /= type_freq.reshape(-1, 1)
        np.save(posttype_f, type_posteriors)

        return type_posteriors