import os
import sys

import numpy as np

from eval.ner.readers.brown import prepare_cluster_to_word_map as brown_map
from hmtm import HMTM
from inference.sum_product import SumProduct


__author__ = 'sim'


class HMRTM(HMTM):
    """
    Hidden Markov Relation Tree Model
    """

    def __init__(self, N, M, R=None, params=None, writeout=False, brown_init_path=None, x_dict=None, approx=False,
                 dirname=None, omit_class_cond=False, omit_emis_cond=False):
        """
        :param N: number of states
        :param M: number of observation symbols
        :param R: number of dep relations (relation-specific HMTM only)
        :param params: numpy objects
          -initial_probs
          -transition_probs
          -final_probs
          -emission_probs)
        :param writeout: save hmm details to a file
        :param omit_class_cond: do not condition the class variable on the relation variable
        :param omit_emis_cond: do not condition the output/emission variable on the relation variable
        """
        if dirname is None:
            sys.exit("Output dirname not given.")
        self.dirname = dirname
        self.N = N
        self.start_N = None  # for split-merge
        self.M = M
        self.R = R
        self.omit_class_cond = omit_class_cond
        self.omit_emis_cond = omit_emis_cond
        # initial state probability vector

        if self.omit_class_cond:
            self.initial_probs = np.zeros(N, 'f')
            self.transition_probs = np.zeros([N, N], 'f')
            self.final_probs = np.zeros(N, 'f')
        else:
            self.initial_probs = np.zeros([N, R], 'f')
            self.transition_probs = np.zeros([N, N, R], 'f')
            self.final_probs = np.zeros([N, R], 'f')

        if self.omit_emis_cond:
            self.emission_probs = np.zeros([M, N], 'f')
        else:
            self.emission_probs = np.zeros([M, N, R], 'f')

        self.params_fixed_path = None
        self.params_fixed_type = None  # random init or trained init; set by experimental script
        self.brown_init_path = brown_init_path

        if not params:
            if brown_init_path is None:
                self.initialize_params()
                self.params_exist = False
            else:
                if x_dict is None:
                    sys.exit("wordrep vocab missing")
                self.initialize_brown_params(self.brown_init_path, x_dict, dist_even=True)
                self.params_exist = False

        else:
            try:
                (self.initial_probs,
                 self.transition_probs,
                 self.final_probs,
                 self.emission_probs) = params
                self.initial_probs = self.initial_probs.astype('f', copy=False)
                self.transition_probs = self.transition_probs.astype('f', copy=False)
                self.final_probs = self.final_probs.astype('f', copy=False)
                self.emission_probs = self.emission_probs.astype('f', copy=False)
                self.params_exist = True
            except ValueError:
                print("Number of provided model parameters not right.")

        # for updates in em_multiprocess
        self.total_ll = 0.0
        # Count matrices; use 64 dtype here to avoid overflow
        self.initial_counts = np.zeros([self.N, self.R])
        self.transition_counts = np.zeros([self.N, self.N, self.R])
        self.final_counts = np.zeros([self.N, self.R])
        self.emission_counts = np.zeros([self.M, self.N, self.R])

        # storing log likelihoods per iteration
        self.lls = []

        self.sanity_check_init()
        self.inference = SumProduct(approximate=approx)
        self.max_iter = None
        self.n_proc = None
        self.n_sent = None
        self.data_name = None
        self.data_n_tokens = None
        #online EM:
        self.minibatch_size = None
        self.alpha = None
        self.a = None
        self.permute = None

        self.posttypes = None

        self.hmm_type = None

        self.writeout = writeout

    def sanity_check_init(self, logger=None):
        """ Verify dimensions and column-stochasticness"""
        if self.omit_class_cond:
            assert self.initial_probs.shape == (self.N,)
            assert self.transition_probs.shape == (self.N, self.N)
            assert self.final_probs.shape == (self.N,)
        else:
            assert self.initial_probs.shape == (self.N, self.R)
            assert self.transition_probs.shape == (self.N, self.N, self.R)
            assert self.final_probs.shape == (self.N, self.R)

        if self.omit_emis_cond:
            assert self.emission_probs.shape == (self.M, self.N)
        else:
            assert self.emission_probs.shape == (self.M, self.N, self.R)

        if self.omit_class_cond:
            # should be 1 up to some numerical precision:
            assert np.isclose(np.sum(self.initial_probs), 1, atol=1e-02), logger.debug(
                np.sum(self.initial_probs)) if logger is not None else print(np.sum(self.initial_probs))
            # combined transition and final probs must sum to one:
            stacked_probs = np.vstack((self.transition_probs, self.final_probs))
        else:
            for r in range(self.R):
                assert np.isclose(np.sum(self.initial_probs[:, r]), 1, atol=1e-02), logger.debug(
                    np.sum(self.initial_probs[:, r])) if logger is not None else print(np.sum(self.initial_probs[:, r]))
                # combined transition and final probs must sum to one:
                stacked_probs = np.vstack((self.transition_probs[:, :, r], self.final_probs[:, r]))
        assert np.allclose(np.sum(stacked_probs, 0), 1, atol=1e-02), logger.debug(
            np.sum(stacked_probs, 0)) if logger is not None else print(np.sum(stacked_probs, 0))

        if self.omit_emis_cond:
            assert np.allclose(np.sum(self.emission_probs, 0), 1, atol=1e-02), logger.debug(
                np.sum(self.emission_probs, 0)) if logger is not None else print(np.sum(self.emission_probs, 0))
        else:
            for r in range(self.R):
                assert np.allclose(np.sum(self.emission_probs[:, :, r], 0), 1, atol=1e-02), logger.debug(
                    np.sum(self.emission_probs[:, :, r], 0)) if logger is not None else print(
                    np.sum(self.emission_probs[:, :, r], 0))

    def init_rand_params(self):
        if self.omit_class_cond:
            initial_probs = np.random.rand(self.N).astype('f')
            transition_probs = np.random.rand(self.N, self.N).astype('f')
            final_probs = np.random.rand(self.N).astype('f')
        else:
            initial_probs = np.random.rand(self.N, self.R).astype('f')
            transition_probs = np.random.rand(self.N, self.N, self.R).astype('f')
            final_probs = np.random.rand(self.N, self.R).astype('f')

        if self.omit_emis_cond:
            emission_probs = np.random.rand(self.M, self.N).astype('f')
        else:
            emission_probs = np.random.rand(self.M, self.N, self.R).astype('f')

        return initial_probs, transition_probs, final_probs, emission_probs

    def normalize_params(self, initial_probs, transition_probs, final_probs, emission_probs):
        if self.omit_class_cond:
            self.initial_probs = initial_probs / np.sum(initial_probs)
            sums = np.sum(transition_probs, 0) + final_probs  # sum along columns
            self.transition_probs = transition_probs / sums  # sums gets broadcast
            self.final_probs = final_probs / sums
        else:
            self.initial_probs = np.zeros([self.N, self.R], 'f')
            self.transition_probs = np.zeros([self.N, self.N, self.R], 'f')
            self.final_probs = np.zeros([self.N, self.R], 'f')
            for r in range(self.R):
                self.initial_probs[:, r] = initial_probs[:, r] / np.sum(initial_probs[:, r])
                # don't forget to add final_probs to transition_probs
                sums = np.sum(transition_probs[:, :, r], 0) + final_probs[:, r]  # sum along columns
                self.transition_probs[:, :, r] = transition_probs[:, :, r] / sums
                self.final_probs[:, r] = final_probs[:, r] / sums

        if self.omit_emis_cond:
            sums = np.sum(emission_probs, 0)  # sum along columns
            self.emission_probs = emission_probs / sums
        else:
            self.emission_probs = np.zeros([self.M, self.N, self.R], 'f')
            for r in range(self.R):
                sums = np.sum(emission_probs[:, :, r], 0)  # sum along columns
                self.emission_probs[:, :, r] = emission_probs[:, :, r] / sums

    def clear_counts(self, smoothing=1e-8):
        """ Clear the count tables for another iteration.
        Smoothing might be preferred to avoid "RuntimeWarning: divide by zero encountered in log"
        """
        # use 64 dtype here to avoid overflow
        if self.omit_class_cond:
            self.initial_counts = np.zeros(self.N)
            self.transition_counts = np.zeros([self.N, self.N])
            self.final_counts = np.zeros(self.N)
        else:
            self.initial_counts = np.zeros([self.N, self.R])
            self.transition_counts = np.zeros([self.N, self.N, self.R])
            self.final_counts = np.zeros([self.N, self.R])

        if self.omit_emis_cond:
            self.emission_counts = np.zeros([self.M, self.N])
        else:
            self.emission_counts = np.zeros([self.M, self.N, self.R])

        self.initial_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.final_counts.fill(smoothing)
        self.emission_counts.fill(smoothing)

    def treerepr_scores(self, tree):
        """
        Tree-analogue to trellis_scores; potentials depend on the relation

        :param tree: tree graph

        """
        if self.omit_class_cond:
            # every leaf gets initial_probs
            for leaf in tree.get_leaves():
                leaf.set_initial_potentials(np.log(self.initial_probs))
            # every edge gets transition_probs
            for edge in tree.get_edges_not_to_root():
                edge.set_potentials(np.log(self.transition_probs))
            # every edge to # root gets final_probs
            for edge in tree.get_edges_to_root():
                edge.set_potentials(np.log(self.final_probs))
        else:
            # every leaf gets initial_probs
            for leaf in tree.get_leaves():
                leaf.set_initial_potentials(np.log(self.initial_probs[:, leaf.rel]))
            # every edge gets transition_probs
            for edge in tree.get_edges_not_to_root():
                edge.set_potentials(np.log(self.transition_probs[:, :, edge.parent.rel]))
            # every edge to # root gets final_probs
            for edge in tree.get_edges_to_root():
                edge.set_potentials(
                    np.log(self.final_probs[:, edge.child.rel]))  # because trans and final probs are tied (
                # should sum to 1 columwise when stacked, we have final probs conditioned on child's rel

        if self.omit_emis_cond:
            # every node except root gets emission_probs
            for node in tree.get_nonroots():
                node.set_potentials(np.log(self.emission_probs[node.get_name(), :]))
        else:
            # every node except root gets emission_probs
            for node in tree.get_nonroots():
                node.set_potentials(np.log(self.emission_probs[node.get_name(), :, node.rel]))

    def update_counts_from_tree(self, tree):
        """
        In E-step:
        Update the count matrices with partials from one tree

        BUG: can overflow because of the large log posteriors in the case of a huge tree
        get extremely big when taking exp
        TODO: fix by postponing the exp from compute_posteriors() until compute_parameters()
        """
        if self.omit_class_cond:
            self.initial_counts += sum([leaf.posterior for leaf in tree.get_leaves()])
            for edge in tree.get_edges_not_to_root():
                self.transition_counts += edge.posterior
            self.final_counts += sum([edge.posterior for edge in tree.get_edges_to_root()])
        else:
            for leaf in tree.get_leaves():
                self.initial_counts[:, leaf.rel] += leaf.posterior
            for edge in tree.get_edges_not_to_root():
                self.transition_counts[:, :, edge.parent.rel] += edge.posterior
            for edge in tree.get_edges_to_root():
                self.final_counts[:, edge.child.rel] += edge.posterior

        if self.omit_emis_cond:
            for node in tree.get_nonroots():
                self.emission_counts[node.get_name(), :] += node.posterior
        else:
            for node in tree.get_nonroots():
                self.emission_counts[node.get_name(), :, node.rel] += node.posterior

    def compute_online_parameters(self, t):
        """
        In M-step of online EM: normalize the counts; interpolate between the old parameters
        and the contribution of new probs.
        (1-eta_t)*param^(t-1) + eta_t*probs

        Note: different from Liang and Klein 2009, and Cappe 2009 in that we interpolate probs directly

        Doesn't exploit the sparsity of the counts.

        :param t: minibatch (update) number
        """
        # stepsize
        eta = self.compute_eta(t)
        assert not np.isnan(self.initial_counts.sum())
        assert not np.isnan(self.transition_counts.sum())
        assert not np.isnan(self.emission_counts.sum())
        assert not np.isnan(self.final_counts.sum())

        if self.omit_class_cond:
            self.initial_probs = (
                (1 - eta) * self.initial_probs + eta * (self.initial_counts / np.sum(self.initial_counts))).astype('f')
            sums = np.sum(self.transition_counts, 0) + self.final_counts
            self.transition_probs = ((1 - eta) * self.transition_probs + eta * (self.transition_counts / sums)).astype(
                'f')
            self.final_probs = ((1 - eta) * self.final_probs + eta * (self.final_counts / sums)).astype('f')
        else:
            for r in range(self.R):
                self.initial_probs[:, r] = ((1 - eta) * self.initial_probs[:, r] + eta * (
                    self.initial_counts[:, r] / np.sum(self.initial_counts[:, r]))).astype('f')
                sums = np.sum(self.transition_counts[:, :, r], 0) + self.final_counts[:, r]
                self.transition_probs[:, :, r] = (
                    (1 - eta) * self.transition_probs[:, :, r] + eta * (self.transition_counts[:, :, r] / sums)).astype(
                    'f')
                self.final_probs[:, r] = (
                    (1 - eta) * self.final_probs[:, r] + eta * (self.final_counts[:, r] / sums)).astype('f')

        if self.omit_emis_cond:
            self.emission_probs = (
                (1 - eta) * self.emission_probs + eta * (
                    self.emission_counts / np.sum(self.emission_counts, 0))).astype('f')
        else:
            for r in range(self.R):
                self.emission_probs[:, :, r] = ((1 - eta) * self.emission_probs[:, :, r] + eta * (
                    self.emission_counts[:, :, r] / np.sum(self.emission_counts[:, :, r], 0))).astype('f')

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

                if self.omit_class_cond:
                    initial_counts += sum([leaf.posterior for leaf in tree.get_leaves()])
                    for edge in tree.get_edges_not_to_root():
                        transition_counts += edge.posterior
                    final_counts += sum([edge.posterior for edge in tree.get_edges_to_root()])
                else:
                    for leaf in tree.get_leaves():
                        initial_counts[:, leaf.rel] += leaf.posterior
                    for edge in tree.get_edges_not_to_root():
                        transition_counts[:, :, edge.parent.rel] += edge.posterior
                    for edge in tree.get_edges_to_root():
                        final_counts[:, edge.child.rel] += edge.posterior

                if self.omit_emis_cond:
                    for node in tree.get_nonroots():
                        emission_counts[node.get_name(), :] += node.posterior
                else:
                    for node in tree.get_nonroots():
                        emission_counts[node.get_name(), :, node.rel] += node.posterior

                total_ll += tree.get_ll()

                tree.clear_tree()

            return initial_counts, transition_counts, final_counts, emission_counts, total_ll
        except KeyboardInterrupt:
            pass

    def compute_parameters(self, logger):
        """
        In M-step: normalize the counts to obtain true parameters.
        """
        if logger is not None:
            logger.info("Recomputing parameters.")

        if self.omit_class_cond:
            self.initial_probs = (self.initial_counts / np.sum(self.initial_counts)).astype(
                'f')  # probs should be 32 dtype
            sums = np.sum(self.transition_counts, 0) + self.final_counts
            self.transition_probs = (self.transition_counts / sums).astype('f')
            self.final_probs = (self.final_counts / sums).astype('f')
        else:
            for r in range(self.R):
                self.initial_probs[:, r] = (self.initial_counts[:, r] / np.sum(self.initial_counts[:, r])).astype(
                    'f')  # probs should be 32 dtype
                sums = np.sum(self.transition_counts[:, :, r], 0) + self.final_counts[:, r]
                self.transition_probs[:, :, r] = (self.transition_counts[:, :, r] / sums).astype('f')
                self.final_probs[:, r] = (self.final_counts[:, r] / sums).astype('f')

        if self.omit_emis_cond:
            self.emission_probs = (self.emission_counts / np.sum(self.emission_counts, 0)).astype('f')
        else:
            for r in range(self.R):
                self.emission_probs[:, :, r] = (
                    self.emission_counts[:, :, r] / np.sum(self.emission_counts[:, :, r], 0)).astype('f')

    def split_params(self, noise_amount):
        """
         Split states in two. Each state parameters are copied and some noise added.
        """
        split_dim = self.N * 2

        if self.omit_class_cond:
            initial_probs_split = self.initial_probs.repeat(2, axis=0)  # split along columns
            r = np.random.normal(0, noise_amount, initial_probs_split.shape)  # noise
            initial_probs_split += initial_probs_split * r  # downscale r according to individual values in initial_probs...

            transition_probs_split = self.transition_probs.repeat(2, axis=1).repeat(2,
                                                                                    axis=0)  # split along columns then rows
            r = np.random.normal(0, noise_amount, transition_probs_split.shape)  # noise
            transition_probs_split += transition_probs_split * r  # downscale r according to individual values in initial_probs...

            final_probs_split = self.final_probs.repeat(2, axis=0)  # split along columns
            r = np.random.normal(0, noise_amount, final_probs_split.shape)  # noise
            final_probs_split += final_probs_split * r  # downscale r according to individual values in initial_probs...
        else:
            initial_probs_split = np.zeros([split_dim, self.R], 'f')
            transition_probs_split = np.zeros([split_dim, split_dim, self.R], 'f')
            final_probs_split = np.zeros([split_dim, self.R], 'f')
            for rel in range(self.R):
                initial_probs_split[:, rel] = self.initial_probs[:, rel].repeat(2, axis=0)  # split along columns
                r = np.random.normal(0, noise_amount, initial_probs_split[:, rel].shape)  # noise
                initial_probs_split[:, rel] += initial_probs_split[:,
                                               rel] * r  # downscale r according to individual values in initial_probs...
                transition_probs_split[:, :, rel] = self.transition_probs[:, :, rel].repeat(2, axis=1).repeat(2,
                                                                                                              axis=0)  # split along columns then rows
                r = np.random.normal(0, noise_amount, transition_probs_split[:, :, rel].shape)  # noise
                transition_probs_split[:, :, rel] += transition_probs_split[:, :,
                                                     rel] * r  # downscale r according to individual values in initial_probs...
                final_probs_split[:, rel] = self.final_probs[:, rel].repeat(2, axis=0)  # split along columns
                r = np.random.normal(0, noise_amount, final_probs_split[:, rel].shape)  # noise
                final_probs_split[:, rel] += final_probs_split[:,
                                             rel] * r  # downscale r according to individual values in initial_probs...

        if self.omit_emis_cond:
            emission_probs_split = self.emission_probs.repeat(2, axis=1)  # split along columns
            r = np.random.normal(0, noise_amount, emission_probs_split.shape)  # noise
            emission_probs_split += emission_probs_split * r  # downscale r according to individual values in initial_probs...
        else:
            emission_probs_split = np.zeros([self.M, split_dim, self.R], 'f')
            for rel in range(self.R):
                emission_probs_split[:, :, rel] = self.emission_probs[:, :, rel].repeat(2,
                                                                                        axis=1)  # split along columns
                r = np.random.normal(0, noise_amount, emission_probs_split[:, :, rel].shape)  # noise
                emission_probs_split[:, :, rel] += emission_probs_split[:, :,
                                                   rel] * r  # downscale r according to individual values in initial_probs...

        assert initial_probs_split.shape[0] == final_probs_split.shape[0] == transition_probs_split.shape[0] == \
               emission_probs_split.shape[1]

        self.N = transition_probs_split.shape[0]

        return initial_probs_split, transition_probs_split, final_probs_split, emission_probs_split

    def initialize_brown_params(self, brown_init_path, x_dict, c_factor=1000, dist_even=True):
        """ init parameters to be non-random column-stochastic matrices
        based on brown clusters
        Concerns emission params only (for now) although transitions could be approximated somehow as well.

        Assume for now that n of clusters = state size.
        Some words might not be found in clusters.

        For w belonging to cluster c_x, we put most of the prob mass to w|c_x, and distribute remaining prob mass
        unevenly or evenly among all other c_y.

        First initialize randomly, then all w entries belonging to c_x are multiplied by c_factor; finally, normalize.
        """
        initial_probs, transition_probs, final_probs, emission_probs = self.init_rand_params()
        c_to_w = brown_map(brown_init_path)
        assert len(c_to_w) == self.N
        if dist_even:
            if self.omit_emis_cond:
                emission_probs = np.zeros((self.M, self.N)).astype('f') + np.random.rand()
            else:
                emission_probs = np.zeros((self.M, self.N, self.R)).astype('f') + np.random.rand()

        for c, c_id in enumerate(c_to_w):
            w_ids = self.get_label_ids(c_to_w[c_id], x_dict)  # x_dict.get_label_name(w_id)
            if self.omit_emis_cond:
                emission_probs[w_ids, c] *= c_factor
            else:
                for r in range(self.R):
                    emission_probs[w_ids, c, r] *= c_factor

        self.normalize_params(initial_probs, transition_probs, final_probs, emission_probs)

    def posterior_cont_type_decode_corpus(self, dataset, rep_dataset, logger=None, ignore_rel=None):
        """Run posterior_decode at corpus level,
        return continuous rep per type (avg. over posteriors in all
        instances). """
        if self.posttypes is None:
            if self.dirname is not None:
                assert len(dataset.wordrep_dict) == len(rep_dataset.x_dict)
                posttype_f = "{}posttype{}.npy".format(self.dirname, ignore_rel or "")
                self.posttypes = np.load(posttype_f) if os.path.exists(posttype_f) else self.obtain_posttypes(
                    posttype_f, rep_dataset, len(dataset.wordrep_dict), logger=logger, ignore_rel=ignore_rel)
                assert self.posttypes.shape == (len(dataset.wordrep_dict), self.N, self.R)
            else:
                sys.exit("dirname not set properly")

        if logger is not None: logger.info("Decoding on eval datasets.")
        # assign posteriors to types in dataset
        for seq in dataset.seq_list:
            if seq.t is None:
                print("seq.t is None")
                seq.u = None
                continue
            seq.u = {}
            for node in seq.t.get_nonroots():
                post = self.posttypes[node.name, :, node.rel]
                if not np.isnan(
                        np.sum(post)) and node.rel is not ignore_rel:  # second check probably redundant as isnan anyway
                    seq.u[node.index] = post
            seq.t = None

    def obtain_posttypes_cumul(self, posttype_f, rep_dataset, n_types, logger=None, ignore_rel=None):
        super().obtain_posttypes(posttype_f=posttype_f, rep_dataset=rep_dataset, n_types=n_types, logger=logger)

    def obtain_posttypes(self, posttype_f, rep_dataset, n_types, logger=None, ignore_rel=None):
        if logger is not None: logger.info("Obtaining posterior type counts.")
        # obtain type posteriors
        type_posteriors = np.zeros((n_types, self.N, self.R))
        type_freq = np.zeros((n_types, self.R))
        for count, tree in enumerate(rep_dataset.train):
            # posteriors is dict with keys starting at 1
            if tree is None:
                # print("tree is None")
                continue
            if logger is not None:
                if count % 1000 == 0:
                    logger.debug(count)
            posteriors = self.posterior_decode(tree, cont=True, ignore_rel=ignore_rel)
            for node in tree.get_nonroots():
                if node.index in posteriors:
                    type_posteriors[node.name, :, node.rel] += posteriors[node.index]
                    type_freq[node.name, node.rel] += 1
        # normalize
        for r in range(self.R):
            type_posteriors[:, :, r] /= type_freq[:, r].reshape(-1, 1)  # yields NaNs, avoided by the parent method
        np.save(posttype_f, type_posteriors)

        return type_posteriors

    def write_add(self, out):
        out.write("Number of relations: {}\n".format(self.R))
        out.write("Omit class conditioning: {}\n".format(self.omit_class_cond))
        out.write("Omit emis conditioning: {}\n".format(self.omit_emis_cond))