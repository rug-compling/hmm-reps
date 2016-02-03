import logging
import shutil
import sys

from eval.ner.lxmls.readers.brown import prepare_cluster_to_word_map as brown_map
import inference.hmm_inference as infer
from inference.hmm_splitmerge import HMMsplitmerge
from output.emission_prob import get_best_clusterids
from util.log_domain import *
from util.util import *


def prepare_dirname(hmm_type="", append_string="", lang="", max_iter=0, N=0, n_sent=0, alpha=0, minibatch_size=0):
    dirname = "hmm_{}_{}_miter{}_N{}_nsent{}_alpha{}_batchsize{}_{}".format(lang, hmm_type, max_iter, N,
                                                                            n_sent, alpha,
                                                                            minibatch_size, append_string)
    if os.path.exists(dirname):
        print("Experiment output directory already exists. Appending with '_temp'.")
        dirname = "{}{}".format(dirname, "_temp")
        if os.path.exists(dirname):  # remove temp directory
            print("Removing _temp directory.")
            shutil.rmtree("{}".format(dirname))
        os.makedirs(dirname)
    else:
        os.makedirs(dirname)

    return dirname


class HMM:
    def __init__(self, N, M, params=None, writeout=False, brown_init_path=None, x_dict=None, approx=False,
                 dirname=None):

        """
        :param N: number of states
        :param M: number of observation symbols
        :param params: numpy objects
          -initial_probs
          -transition_probs
          -final_probs
          -emission_probs)
        :param writeout: save hmm details to a file
        """
        if dirname is None:
            sys.exit("Output dirname not given.")
        self.dirname = dirname
        self.N = N
        self.start_N = None  # for split-merge
        self.M = M

        # initial state probability vector

        self.initial_probs = np.zeros(N, 'f')
        # transition matrix_ij[NxN], p(i=current state | j=previous state)
        self.transition_probs = np.zeros([N, N], 'f')
        # final state probability vector
        self.final_probs = np.zeros(N, 'f')
        # emission matrix_kj[MxN], p(k=current observation | j=current state)
        self.emission_probs = np.zeros([M, N], 'f')

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

        #for updates in em_multiprocess
        self.total_ll = 0.0
        #Count matrices; use 64 dtype here to avoid overflow
        self.initial_counts = np.zeros(self.N)
        self.transition_counts = np.zeros([self.N, self.N])
        self.final_counts = np.zeros(self.N)
        self.emission_counts = np.zeros([self.M, self.N])

        #storing log likelihoods per iteration
        self.lls = []

        self.sanity_check_init()
        self.inference = infer.HMMInference(approximate=approx)
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

    def setup_logging(self, level):
        """
        :param level: logging.INFO or logging.DEBUG, etc.
        """
        #logfile = "{}/hmm.log".format(self.dirname)
        #logging.basicConfig(filename=logfile, level=level,
        #                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logging.captureWarnings(True)

        return logging.getLogger(__name__)

    def prepare_logging(self, dataset, max_iter, n_proc, logging_level, noise_amount=None, sensitivity=None,
                        minibatch_size=None, alpha=None, a=None, permute=None, hmm_type="", append_string=""):
        self.max_iter = max_iter
        self.n_proc = n_proc
        self.n_sent = dataset.howbig
        self.data_name = dataset.corpus_file
        try:
            self.data_n_tokens = dataset.train.get_num_tokens()
        except AttributeError:
            self.data_n_tokens = None
        self.noise_amount = noise_amount
        self.sensitivity = sensitivity
        self.minibatch_size = minibatch_size
        self.hmm_type = hmm_type
        self.alpha = alpha
        self.a = a
        self.permute = permute
        # self.prepare_dirname(hmm_type, append_string)
        logger = self.setup_logging(logging_level)
        logger.info("Initialized.")
        if self.writeout: self.write_initialized_probs(logger)
        return logger

    def initialize_params(self):
        """ init parameters to be random column-stochastic matrices """
        initial_probs, transition_probs, final_probs, emission_probs = self.init_rand_params()
        self.normalize_params(initial_probs, transition_probs, final_probs, emission_probs)

    def init_rand_params(self):
        initial_probs = np.random.rand(self.N).astype('f')
        transition_probs = np.random.rand(self.N, self.N).astype('f')
        final_probs = np.random.rand(self.N).astype('f')
        emission_probs = np.random.rand(self.M, self.N).astype('f')

        return initial_probs, transition_probs, final_probs, emission_probs

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
            emission_probs = np.zeros((self.M, self.N)).astype('f') + np.random.rand()
        for c, c_id in enumerate(c_to_w):
            w_ids = self.get_label_ids(c_to_w[c_id], x_dict)  # x_dict.get_label_name(w_id)
            emission_probs[w_ids, c] *= c_factor

        self.normalize_params(initial_probs, transition_probs, final_probs, emission_probs)

    def get_label_ids(self, words, x_dict):
        return [x_dict.get_label_id(w) for w in words if w in x_dict]

    def split_params(self, noise_amount):
        """
         Split states in two. Each state parameters are copied and some noise added.
        """
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

        emission_probs_split = self.emission_probs.repeat(2, axis=1)  # split along columns
        r = np.random.normal(0, noise_amount, emission_probs_split.shape)  # noise
        emission_probs_split += emission_probs_split * r  # downscale r according to individual values in initial_probs...

        assert initial_probs_split.shape[0] == final_probs_split.shape[0] == transition_probs_split.shape[0] == \
               emission_probs_split.shape[1]

        self.N = transition_probs_split.shape[0]

        return initial_probs_split, transition_probs_split, final_probs_split, emission_probs_split

    def normalize_params(self, initial_probs, transition_probs, final_probs, emission_probs):
        self.initial_probs = initial_probs / np.sum(initial_probs)
        # don't forget to add final_probs to transition_probs
        sums = np.sum(transition_probs, 0) + final_probs  # sum along columns
        self.transition_probs = transition_probs / sums  # sums gets broadcast
        self.final_probs = final_probs / sums
        sums = np.sum(emission_probs, 0)  # sum along columns
        self.emission_probs = emission_probs / sums

    def sanity_check_init(self, logger=None):
        """ Verify dimensions and column-stochasticness"""
        assert self.initial_probs.shape == (self.N,)
        assert self.transition_probs.shape == (self.N, self.N)
        assert self.final_probs.shape == (self.N,)
        assert self.emission_probs.shape == (self.M, self.N)

        # should be 1 up to some numerical precision:
        assert np.isclose(np.sum(self.initial_probs), 1, atol=1e-02), logger.debug(
            np.sum(self.initial_probs)) if logger is not None else print(np.sum(self.initial_probs))
        assert np.allclose(np.sum(self.emission_probs, 0), 1, atol=1e-02), logger.debug(
            np.sum(self.emission_probs, 0)) if logger is not None else print(np.sum(self.emission_probs, 0))
        # combined transition and final probs must sum to one:
        stacked_probs = np.vstack((self.transition_probs, self.final_probs))
        assert np.allclose(np.sum(stacked_probs, 0), 1, atol=1e-02), logger.debug(
            np.sum(stacked_probs, 0)) if logger is not None else print(np.sum(stacked_probs, 0))

    def em(self, dataset, max_iter=5, logging_level=logging.DEBUG, hmm_type="", append_string=""):
        """
        Baum-Welch algorithm, i.e. Expectation-maximization for HMMs.

        :param dataset: SequenceList object
        :param max_iter: maximum number of iterations (ending condition in addition to ll-decrease)
        """

        # for logging
        logger = self.prepare_logging(dataset, max_iter, 1, logging_level, hmm_type=hmm_type,
                                      append_string=append_string)

        self.em_core(dataset, max_iter, logger)

        if self.writeout:
            self.write_parameters(logger)

    def em_core(self, dataset, max_iter, logger):
        for it in range(1, max_iter + 1):
            total_ll = 0.0
            self.clear_counts()
            for c, seq in enumerate(dataset.train.seq_list, 1):
                initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores(seq)
                state_posteriors, transition_posteriors, ll = self.inference.compute_posteriors(initial_scores,
                                                                                                transition_scores,
                                                                                                final_scores,
                                                                                                emission_scores)
                # update counts for that sequence
                self.update_counts(seq, state_posteriors, transition_posteriors)
                total_ll += ll
                if c % 1000 == 0:
                    logger.info("{} sents".format(c))
                if c % 10 == 0:
                    logger.debug("{} sents".format(c))
            logger.info("Iter: {}\tLog-likelihood: {}".format(it, total_ll))
            self.lls.append(total_ll)
            # new parameters with states split
            self.compute_parameters(logger)

    def em_core_online(self, dataset, max_iter, logger):
        for it in range(1, max_iter + 1):
            total_ll = 0.0
            for t, minibatch in enumerate(chunk_seqs_by_size(dataset.train.seq_list, self.minibatch_size, self.permute),
                                          0):
                # E-step
                self.clear_counts()
                for seq in minibatch:
                    # prepare trellis
                    initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores(seq)
                    # inference (obtain gammas (state and transition posteriors)and ll):
                    state_posteriors, transition_posteriors, ll = self.inference.compute_posteriors(initial_scores,
                                                                                                    transition_scores,
                                                                                                    final_scores,
                                                                                                    emission_scores)
                    # update counts for that sequence
                    self.update_counts(seq, state_posteriors, transition_posteriors)
                    total_ll += ll
                # M-step
                self.compute_online_parameters(t)
                logger.info("minibatch {} (size {})".format(t, self.minibatch_size))

            logger.info("Iter: {}\tLog-likelihood: {}".format(it, total_ll))
            self.lls.append(total_ll)

    def em_splitmerge(self, dataset, max_iter=1, desired_n=8, noise_amount=0.001, sensitivity=0.5, minibatch_size=None,
                      alpha=None, a=None, permute=None, n_proc=1, logging_level=logging.INFO, hmm_type="",
                      append_string=""):
        """
        Implenents split-merge training, as in
        Petrov, Slav (2009) Coarse-to-Fine Natural Language Processing,
        as well as simplified state splitting only.

        Number of states is progressively increased with binary splitting, then some splits are reverted.
        Only those splits are rolled back that result in a smaller likelihood loss compared to other splits.
        The parameter estimation is hierarchical.

        For some reason, the incurred losses are strictly smaller for merges of states with higher indices. I've tracked
        the issue down to emission_counts/emission_probs (are used as weights for backward scores); "higher" states are
        weighted much less heavily.

        Setting sensitivity to 0 avoids merging.

        :param desired_n: desired final number of states obtained by splitting
        """
        # for logging
        logger = self.prepare_logging(dataset, max_iter, n_proc, logging_level, noise_amount=noise_amount,
                                      sensitivity=sensitivity, minibatch_size=minibatch_size, alpha=alpha, a=a,
                                      permute=permute, hmm_type=hmm_type, append_string=append_string)

        # type of em
        if minibatch_size is None:
            em_train = self.em_core_multiprocess if self.n_proc > 1 else self.em_core
        else:
            em_train = self.em_core_online_multiprocess if self.n_proc > 1 else self.em_core_online

        # note starting number of states for later ref.
        self.start_N = self.N

        if not self.params_fixed_type:  # if using already learnt params, go directly to splitting
            # initialization: train with starting number of states for some number of iterations
            logger.info("Training on starting number of states.")
            em_train(dataset, max_iter, logger)
            logger.info("Saving parameter matrices.")
            self.write_parameter_matrices(append_string=self.N)
        # split-merge cycles
        while self.N < desired_n:
            # split the states (params) into two
            self.split(noise_amount, logger)

            # re-train
            logger.info("Training on split states.")
            em_train(dataset, max_iter, logger)
            logger.info("Saving parameter matrices.")
            if self.N < desired_n:
                self.write_parameter_matrices(append_string=self.N)

            if sensitivity == 0:  # avoid loss calculation and merging
                continue
            # inference with new parameters (of states split)
            logger.info("Performing inference with states split. Collecting loss.")
            total_loss = self.get_total_loss(dataset)
            logger.debug("Total losses incurred per merge: {}".format(total_loss))

            # merge
            logger.info("Merging states.")
            self.merge(total_loss, logger, sensitivity)  # changes self.N

        logger.info("Finished training.")

        if self.writeout:
            self.write_parameters(logger)

    def get_total_loss(self, dataset):
        """
        Perform inference with learnt split params,
        accumulate loss per possible merge over complete dataset
        """
        assert self.N % 2 == 0
        total_loss = np.zeros(self.N / 2)
        # get normalized emission counts for backward pass weighting
        norm_emission_c = np.log(self.emission_counts.sum(axis=0) / self.emission_counts.sum())
        for c, seq in enumerate(dataset.train.seq_list, 1):
            initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores(seq)
            forward, ll, backward, ll2 = self.inference.compute_fb(initial_scores, transition_scores, final_scores,
                                                                   emission_scores)
            # collect loss for all possible merges
            loss_seq = HMMsplitmerge().get_loss(self.N, forward, backward, ll, norm_emission_c)
            assert loss_seq.size == total_loss.size
            total_loss += loss_seq

        assert np.all(total_loss <= 0)
        return total_loss

    def split(self, noise_amount, logger):
        logger.debug("Checking correctness of parameter matrices: self.sanity_check_init()")
        logger.info("Splitting states (params).")
        initial_probs, transition_probs, final_probs, emission_probs = self.split_params(noise_amount)
        logger.info("Changed number of states to {}.".format(self.N))
        # normalize and assign to object:
        self.normalize_params(initial_probs, transition_probs, final_probs, emission_probs)
        logger.debug("Checking correctness of split parameter matrices: self.sanity_check_init()")
        self.sanity_check_init(logger=logger)
        assert self.N % 2 == 0

    def merge(self, losses, logger, sensitivity=0.5):
        """
        :param losses: loss for each possible merge
        :param sensitivity: percentage of merges to perform
        """
        if len(losses) == 1: return
        # number of merges to do
        n_to_merge = math.floor(len(losses) * sensitivity)
        if n_to_merge == 0: return
        ids_to_merge = losses.argsort()[:n_to_merge]

        # new structures to hold params after merge
        assert len(self.initial_probs) == len(self.final_probs)
        assert len(self.initial_probs) == self.emission_probs.shape[1]
        initial_probs = np.zeros(len(self.initial_probs) - n_to_merge, 'f')
        final_probs = np.zeros(len(self.final_probs) - n_to_merge, 'f')
        # auxiliary structure for merged transition params:
        transition_probs_temp = np.zeros((self.transition_probs.shape[0], self.transition_probs.shape[1] - n_to_merge),
                                         'f')
        transition_probs = np.zeros(
            (self.transition_probs.shape[0] - n_to_merge, self.transition_probs.shape[1] - n_to_merge), 'f')
        emission_probs = np.zeros((self.emission_probs.shape[0], self.emission_probs.shape[1] - n_to_merge), 'f')

        n_merged = 0
        for j in range(emission_probs.shape[1]):
            j_split = j + n_merged
            # merge
            if j_split / 2 in ids_to_merge:
                # sum and normalize
                initial_probs[j] = (self.initial_probs[j_split] + self.initial_probs[j_split + 1])
                final_probs[j] = (self.final_probs[j_split] + self.final_probs[j_split + 1]) / 2
                emission_probs[:, j] = (self.emission_probs[:, j_split] + self.emission_probs[:, j_split + 1]) / 2
                transition_probs_temp[:, j] = (self.transition_probs[:, j_split] + self.transition_probs[:,
                                                                                   j_split + 1]) / 2
                n_merged += 1
            # use split
            else:
                initial_probs[j] = self.initial_probs[j_split]
                final_probs[j] = self.final_probs[j_split]
                emission_probs[:, j] = self.emission_probs[:, j_split]
                transition_probs_temp[:, j] = self.transition_probs[:, j_split]

        # assert np.isclose(np.sum(initial_probs),1, atol=1e-03)
        # assert np.allclose(np.sum(emission_probs,0),1, atol=1e-03)

        # merge transition params along rows
        n_merged = 0
        for i in range(transition_probs.shape[0]):
            i_split = i + n_merged
            # merge
            if i_split / 2 in ids_to_merge:
                transition_probs[i, :] = transition_probs_temp[i_split, :] + transition_probs_temp[i_split + 1, :]
                n_merged += 1
            else:
                transition_probs[i, :] = transition_probs_temp[i_split, :]

        # assert transition_probs.shape == (self.transition_probs.shape[0] - n_to_merge,
        # self.transition_probs.shape[1] - n_to_merge)

        self.N = emission_probs.shape[1]
        logger.info("Changed number of states after merge to: {}".format(self.N))

        self.initial_probs = initial_probs
        self.final_probs = final_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs

        self.sanity_check_init(logger=logger)

    def em_multiprocess(self, dataset, max_iter=50, n_proc=4, logging_level=logging.INFO, hmm_type="",
                        append_string=""):
        # for logging
        logger = self.prepare_logging(dataset, max_iter, n_proc, logging_level, hmm_type=hmm_type,
                                      append_string=append_string)

        self.em_core_multiprocess(dataset, max_iter, logger)

        if self.writeout:
            self.write_parameters(logger)

    def em_core_multiprocess(self, dataset, max_iter, logger):
        """
        Baum-Welch algorithm, i.e. Expectation-maximization for HMMs. Parallelized execution on sentences.

        :param dataset:
        :param max_iter: maximum number of iterations (ending condition in addition to ll-decrease)
        """
        from multiprocessing import Pool

        for it in range(1, max_iter + 1):
            self.total_ll = 0.0
            self.clear_counts()

            if self.hmm_type == "tree" or self.hmm_type == "rel" or self.hmm_type == "lr":
                dataset.prepare_trees()
                sents = dataset.train.tree_list
            else:
                sents = dataset.train.seq_list

            sents_chunked = chunk_seqs(sents, self.n_proc)
            pool = Pool(self.n_proc)
            try:
                # compute counts
                ic_tc_fc_ep_ll = pool.map(self.em_process_multiseq, sents_chunked)
                pool.close()
                pool.join()

                # final sum of counts from processes and update to self
                for subcount in ic_tc_fc_ep_ll:
                    self.initial_counts += subcount[0]
                    self.transition_counts += subcount[1]
                    self.final_counts += subcount[2]
                    self.emission_counts += subcount[3]
                    self.total_ll += subcount[4]
            except KeyboardInterrupt:
                pool.terminate()
                logger.exception("Terminated.")
                break

            logger.info("Iter: {}\tLog-likelihood: {}".format(it, self.total_ll))
            self.lls.append(self.total_ll)

            # assert np.isclose(total_ll, self.total_ll)
            # assert np.allclose(self.initial_counts, self.initial_counts2)
            # assert np.allclose(self.transition_counts, self.transition_counts2)
            #assert np.allclose(self.final_counts, self.final_counts2)
            #assert np.allclose(self.emission_counts, self.emission_counts2)

            #M-step
            logger.info("Recomputing parameters.")
            self.compute_parameters(logger)

    def online_em(self, dataset, max_iter=5, minibatch_size=1000, alpha=0.5, a=4, permute=False,
                  logging_level=logging.INFO, hmm_type="", append_string=""):
        """
        Baum-Welch algorithm, i.e. Expectation-maximization for HMMs.
        Using stepwise mini-batch online EM (Liang and Klein 2009, Cappe and Moulines 2009).

        :param dataset: SequenceList object
        :param max_iter: maximum number of iterations (ending condition in addition to ll-decrease)
        :param minibatch_size: number of data instances after which the update is done
        :param alpha: stepwise reduction power
        :param a: param in eta stepsize (minor role)
        """

        # for logging
        logger = self.prepare_logging(dataset, max_iter, 1, logging_level, minibatch_size=minibatch_size,
                                      alpha=alpha, a=a, permute=permute, hmm_type=hmm_type,
                                      append_string=append_string)

        self.em_core_online(dataset, max_iter, logger)

        if self.writeout:
            self.write_parameters(logger)

    def online_em_multiprocess(self, dataset, max_iter=5, minibatch_size=1000, alpha=0.1, a=4, permute=False, n_proc=4,
                               logging_level=logging.INFO, hmm_type="", append_string=""):
        # for logging
        logger = self.prepare_logging(dataset, max_iter, n_proc, logging_level, minibatch_size=minibatch_size,
                                      alpha=alpha, a=a, permute=permute, hmm_type=hmm_type, append_string=append_string)

        self.em_core_online_multiprocess(dataset, max_iter, logger)

        if self.writeout:
            self.write_parameters(logger)

    def em_core_online_multiprocess(self, dataset, max_iter, logger):
        """
        Baum-Welch algorithm, i.e. Expectation-maximization for HMMs.
        Using stepwise mini-batch online EM (Liang and Klein 2009, Cappe and Moulines 2009).

        NOTE: Multiprocess on the examples within minibatch: only faster for large minibatches (=>1000)  and/or large N

        :param dataset: SequenceList object
        :param max_iter: maximum number of iterations (ending condition in addition to ll-decrease)
        :param minibatch_size: number of data instances after which the update is done
        :param alpha: stepwise reduction power
        :param a: param in eta stepsize (minor role)
        """
        from multiprocessing import Pool

        for it in range(1, max_iter + 1):
            total_ll = 0.0
            sents = dataset.prepare_trees_gen() if self.hmm_type == "tree" or self.hmm_type == "rel" or self.hmm_type == "lr" else dataset.train.seq_list
            # for t, minibatch in enumerate(chunk_seqs_by_size(sents, self.minibatch_size, self.permute), 0):
            for t, minibatch in enumerate(chunk_seqs_by_size_gen(sents, self.minibatch_size), 0):
                # E-step
                self.clear_counts()
                sents_chunked = chunk_seqs(minibatch, self.n_proc)
                pool = Pool(self.n_proc)
                try:
                    # compute counts
                    ic_tc_fc_ep_ll = pool.map(self.em_process_multiseq, sents_chunked)
                    pool.close()
                    pool.join()
                    #final sum of counts from processes and update to self
                    for subcount in ic_tc_fc_ep_ll:
                        self.initial_counts += subcount[0]
                        self.transition_counts += subcount[1]
                        self.final_counts += subcount[2]
                        self.emission_counts += subcount[3]
                        total_ll += subcount[4]
                except KeyboardInterrupt:
                    pool.terminate()
                    logger.exception("Terminated.")
                    break
                # M-step
                self.compute_online_parameters(t)
                logger.info("minibatch {} (size {})".format(t, self.minibatch_size))

            logger.info("Iter: {}\tLog-likelihood: {}".format(it, total_ll))
            self.lls.append(total_ll)

    def em_process_multiseq(self, seqs):
        """
        Makes a local copy of count matrices, the worker updates them for all seqs, and finally returns them as yet another partial
        counts.
        """
        try:
            total_ll = 0
            initial_counts = self.initial_counts
            transition_counts = self.transition_counts
            final_counts = self.final_counts
            emission_counts = self.emission_counts

            c = 0
            for c, seq in enumerate(seqs, 1):
                # prepare trellis
                initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores(seq)
                # inference (obtain gammas (state and transition posteriors)and ll):
                state_posteriors, transition_posteriors, ll = self.inference.compute_posteriors(initial_scores,
                                                                                                transition_scores,
                                                                                                final_scores,
                                                                                                emission_scores)

                length = len(seq.x)
                initial_counts += state_posteriors[0, :]
                for pos in range(length):
                    x = seq.x[pos]
                    emission_counts[x, :] += state_posteriors[pos, :]
                transition_counts += transition_posteriors.sum(axis=0)
                final_counts += state_posteriors[length - 1, :]
                total_ll += ll

            return initial_counts, transition_counts, final_counts, emission_counts, total_ll

        except KeyboardInterrupt:
            pass

    def clear_counts(self, smoothing=1e-8):
        """ Clear the count tables for another iteration.
        Smoothing might be preferred to avoid "RuntimeWarning: divide by zero encountered in log"
        """
        # use 64 dtype here to avoid overflow
        self.initial_counts = np.zeros(self.N)
        self.transition_counts = np.zeros([self.N, self.N])
        self.final_counts = np.zeros(self.N)
        self.emission_counts = np.zeros([self.M, self.N])
        self.initial_counts.fill(smoothing)
        self.transition_counts.fill(smoothing)
        self.final_counts.fill(smoothing)
        self.emission_counts.fill(smoothing)

    def trellis_scores(self, seq):
        """ Trellis (matrices) for the sequence that includes logs of
        initial, final, transition and emission probabilities (scores).

        Representation useful for running forward-backward.
        """
        length = len(seq.x)
        N = self.N
        # Initial position
        initial_scores = np.log(self.initial_probs)

        # Intermediate position
        emission_scores = np.zeros([length, N], 'f') + logzero()
        # for each position have one transition matrix:
        transition_scores = np.zeros([length - 1, N, N], 'f') + logzero()
        for pos in range(length):
            # choose only relevant emissions
            emission_scores[pos, :] = np.log(self.emission_probs[seq.x[pos], :])
            if pos > 0:
                transition_scores[pos - 1, :, :] = np.log(self.transition_probs)

        # Final position
        final_scores = np.log(self.final_probs)

        return initial_scores, transition_scores, final_scores, emission_scores

    def trellis_scores_decoding(self, seq):
        """ Trellis (matrices) for the sequence that includes logs of
        initial, final, transition and emission probabilities (scores).

        Representation useful for decoding with seq having extra representation (*unk*).
        """
        length = len(seq.z)
        N = self.N
        # Initial position
        initial_scores = np.log(self.initial_probs)

        # Intermediate position
        emission_scores = np.zeros([length, N], 'f') + logzero()
        # for each position have one transition matrix:
        transition_scores = np.zeros([length - 1, N, N], 'f') + logzero()
        for pos in range(length):
            # choose only relevant emissions
            emission_scores[pos, :] = np.log(self.emission_probs[seq.z[pos], :])
            if pos > 0:
                transition_scores[pos - 1, :, :] = np.log(self.transition_probs)

        # Final position
        final_scores = np.log(self.final_probs)

        return initial_scores, transition_scores, final_scores, emission_scores

    def update_counts(self, seq, state_posteriors, transition_posteriors):
        """
        In E-step:
        Update the count matrices with partials from one sequence.
        """

        length = len(seq.x)

        assert not np.isnan(state_posteriors.sum())
        assert not np.isnan(transition_posteriors.sum())

        # vectorized:
        self.initial_counts += state_posteriors[0, :]
        # assert np.all(self.initial_counts == self.initial_counts2)

        # vectorized:
        for pos in range(length):
            x = seq.x[pos]
            self.emission_counts[x, :] += state_posteriors[pos, :]
        self.transition_counts += transition_posteriors.sum(axis=0)
        assert not np.isnan(self.emission_counts.sum())
        assert not np.isnan(self.transition_counts.sum())

        #vectorized
        self.final_counts += state_posteriors[length - 1, :]
        assert not np.isnan(self.final_counts.sum())

    def compute_parameters(self, logger):
        """
        In M-step: normalize the counts to obtain true parameters.
        """
        if logger is not None:
            logger.info("Recomputing parameters.")
        self.initial_probs = (self.initial_counts / np.sum(self.initial_counts)).astype('f')  # probs should be 32 dtype

        # don't forget to add final_probs to transition_probs
        sums = np.sum(self.transition_counts, 0) + self.final_counts
        self.transition_probs = (self.transition_counts / sums).astype('f')
        self.final_probs = (self.final_counts / sums).astype('f')

        self.emission_probs = (self.emission_counts / np.sum(self.emission_counts, 0)).astype('f')

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
        self.initial_probs = (
            (1 - eta) * self.initial_probs + eta * (self.initial_counts / np.sum(self.initial_counts))).astype('f')

        # don't forget to add final_probs to transition_probs
        sums = np.sum(self.transition_counts, 0) + self.final_counts
        self.transition_probs = ((1 - eta) * self.transition_probs + eta * (self.transition_counts / sums)).astype('f')
        self.final_probs = ((1 - eta) * self.final_probs + eta * (self.final_counts / sums)).astype('f')

        self.emission_probs = (
            (1 - eta) * self.emission_probs + eta * (self.emission_counts / np.sum(self.emission_counts, 0))).astype(
            'f')

    def compute_eta(self, t):
        """
        stepsize in online EM
        :param t: minibatch (update) number
        """
        eta = (t + self.a) ** (-self.alpha)
        assert eta > 0
        assert eta > (t + 1 + self.a) ** (-self.alpha)

        return eta

    def viterbi_decode(self, seq):
        """
        Compute the most likely sequence of states given the observations,
        by running the Viterbi algorithm.
        """
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores_decoding(seq)
        # Run viterbi
        length, N = emission_scores.shape
        best_states, total_score = self.inference.run_viterbi(initial_scores,
                                                              transition_scores,
                                                              final_scores,
                                                              emission_scores,
                                                              length,
                                                              N)

        return best_states, total_score

    def max_emission_decode(self, seq):
        """
        The most likely state acc. to maximum emission decoding is the state
        that is most likely to emit a given word.
        A sequence is just an independent concatenation of states.
        """
        length = len(seq.z)
        best_states = -np.ones(length, dtype=int)
        for pos in range(length):
            state_ids = get_best_clusterids(self.emission_probs, seq.z[pos], n=1, prob_thresh=0)
            best_states[pos] = state_ids[0]

        return best_states

    def posterior_decode(self, seq, cont):
        """Compute the sequence of states that are individually the most
        probable, given the observations. This is done by maximizing
        the state posteriors, which are computed with the forward-backward
        algorithm."""
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores_decoding(seq)

        state_posteriors = self.inference.compute_state_posteriors(initial_scores,
                                                                   transition_scores,
                                                                   final_scores,
                                                                   emission_scores)
        # continuous
        if cont:
            return np.exp(state_posteriors)
        # discrete
        else:
            return np.argmax(state_posteriors, axis=1)

    def posterior_decode_train(self, seq, cont):
        """
        On wordrep trainset.

        Compute the sequence of states that are individually the most
        probable, given the observations. This is done by maximizing
        the state posteriors, which are computed with the forward-backward
        algorithm."""
        # Compute scores given the observation sequence.
        initial_scores, transition_scores, final_scores, emission_scores = self.trellis_scores(seq)

        state_posteriors = self.inference.compute_state_posteriors(initial_scores,
                                                                   transition_scores,
                                                                   final_scores,
                                                                   emission_scores)
        # continuous
        if cont:
            return np.exp(state_posteriors)
        # discrete
        else:
            return np.argmax(state_posteriors, axis=1)

    def viterbi_decode_corpus(self, dataset):
        """Run viterbi_decode at corpus level."""
        for c, seq in enumerate(dataset.seq_list):
            best_states, _ = self.viterbi_decode(seq)
            seq.w = best_states

    def max_emission_decode_corpus(self, dataset):
        """
        Run maximum emission decoding at corpus level.
        """
        for c, seq in enumerate(dataset.seq_list):
            best_states = self.max_emission_decode(seq)
            seq.w = best_states

    def posterior_decode_corpus(self, dataset):
        """Run posterior_decode at corpus level."""
        for c, seq in enumerate(dataset.seq_list):
            seq.w = self.posterior_decode(seq, cont=False)

    def posterior_cont_decode_corpus(self, dataset):
        """Run posterior_decode at corpus level,
        return continuous rep"""
        for c, seq in enumerate(dataset.seq_list):
            seq.w = self.posterior_decode(seq, cont=True)

    def posterior_cont_type_decode_corpus(self, dataset, rep_dataset, logger=None):
        """Run posterior_decode at corpus level,
        return continuous rep per type (avg. over posteriors in all
        instances). """
        if self.posttypes is None:
            if self.dirname is not None:
                posttype_f = "{}posttype.npy".format(self.dirname)
                # obtain from wordrep trainset
                assert len(dataset.wordrep_dict) == len(rep_dataset.x_dict)
                assert dataset.wordrep_dict == rep_dataset.x_dict
                self.posttypes = np.load(posttype_f) if os.path.exists(posttype_f) else self.obtain_posttypes(
                    posttype_f, rep_dataset, len(dataset.wordrep_dict), logger=logger)
                assert self.posttypes.shape == (len(dataset.wordrep_dict), self.N)
            else:
                sys.exit("dirname not set properly")

        if logger is not None: logger.info("Decoding on eval datasets.")
        # assign posteriors to types in eval dataset
        for seq in dataset.seq_list:
            seq.w = self.posttypes[seq.z]

    def write_initialized_probs(self, logger):
        """
        Save parameters used as initialization before 1st run of EM, for replication
        """

        logger.info("Writing initialized matrices.")

        # initial probabilities
        outfile_ip = "{}/ip_init".format(self.dirname)
        # transition probabilities
        outfile_tp = "{}/tp_init".format(self.dirname)
        # final probabilities
        outfile_fp = "{}/fp_init".format(self.dirname)
        # emission probabilities
        outfile_ep = "{}/ep_init".format(self.dirname)
        np.save(outfile_ip, self.initial_probs)
        np.save(outfile_tp, self.transition_probs)
        np.save(outfile_fp, self.final_probs)
        np.save(outfile_ep, self.emission_probs)

    def write_parameters(self, logger):
        """
        Save parameters to binary file with numpy save (.npy)
        Produce report with experimental details
        """
        logger.info("Writing settings and parameters.")

        with open("{}/settings".format(self.dirname), "w") as out:
            self.write_core(out)
            self.write_add(out)
            self.write_ll(out)
        self.write_parameter_matrices()

    def write_add(self, out):
        pass

    def write_ll(self, out):
        out.write("\nLikelihood per iteration:\n")
        for ll in self.lls:
            out.write("{}\n".format(ll))

    def write_core(self, out):
        out.write("Maximum number of iterations: {}\n".format(self.max_iter))
        out.write("Number of states: {}\n".format(self.N))
        out.write("Number of observation symbols: {}\n".format(self.M))
        out.write("Number of sentences: {}\n".format(self.n_sent))
        out.write("Number of tokens: {}\n".format(self.data_n_tokens))
        out.write("Name of the corpus file: {}\n".format(self.data_name))
        out.write("Number of processes: {}\n".format(self.n_proc))
        # using existing model; if yes, it can be a random or trained initialization
        out.write("Existing matrices used for initialization: {}\n".format(
            "Yes: {}, {}".format(self.params_fixed_path,
                                 self.params_fixed_type) if self.params_exist else "No: {}".format(
                self.brown_init_path)))
        out.write("\nOnline EM settings (None if not applicable)>>>\n")
        out.write("Minibatch size: {}\n".format(self.minibatch_size))
        out.write("Stepwise reduction power 'alpha': {}\n".format(self.alpha))
        out.write("Parameter 'a' value in 'alpha': {}\n".format(self.a))
        out.write("Permutation of minibatches before iteration: {}\n".format(self.permute))
        out.write("\nSplit-merge EM settings (None if not applicable)>>>\n")
        out.write("Starting number of states: {}\n".format(self.start_N))
        out.write("Noise amount: {}\n".format(self.noise_amount))
        out.write("Sensitivity: {}\n".format(self.sensitivity))
        out.write(
            "Approximate inference through vector projection: {}\n".format(self.inference.approximate))

    def write_parameter_matrices(self, append_string=""):
        # initial probabilities
        outfile_ip = "{}/ip{}".format(self.dirname, append_string)
        # transition probabilities
        outfile_tp = "{}/tp{}".format(self.dirname, append_string)
        # final probabilities
        outfile_fp = "{}/fp{}".format(self.dirname, append_string)
        # emission probabilities
        outfile_ep = "{}/ep{}".format(self.dirname, append_string)
        np.save(outfile_ip, self.initial_probs)
        np.save(outfile_tp, self.transition_probs)
        np.save(outfile_fp, self.final_probs)
        np.save(outfile_ep, self.emission_probs)

    def obtain_posttypes(self, posttype_f, rep_dataset, n_types, logger=None):
        if logger is not None: logger.info("Obtaining posterior type counts.")
        # obtain type posteriors
        type_posteriors = np.zeros((n_types, self.N))
        type_freq = np.zeros(n_types)
        for count, seq in enumerate(rep_dataset.train.seq_list, 1):
            if logger is not None:
                if count % 1000 == 0:
                    logger.debug(count)
            posteriors = self.posterior_decode_train(seq, cont=True)
            for c, w_id in enumerate(seq.x):
                type_posteriors[w_id] += posteriors[c]
                type_freq[w_id] += 1
        # normalize
        type_posteriors /= type_freq.reshape(-1, 1)
        np.save(posttype_f, type_posteriors)

        return type_posteriors


