import sys

import numpy as np
from scipy.linalg import norm
from scipy.stats import variation
from eval.ner.lxmls.readers.word2vec import write_embed

from readers.conll_corpus import ConllCorpus
from readers.text_corpus import TextCorpus

from sequences.label_dictionary import *
from readers.vocab import read_vocab
from util.util import nparr_to_str


def compare_sparsity(nparr1, nparr2, sort=True, normalized=True):
    """
    For two two-dimensional arrays, compare their sparsity, i.e. how deterministic they are:
    this is defined as the std. deviation for each row (dimension), or as std. dev. normalized by the mean (variation
    coefficient)

    This can be used to measure sparsity of transitions in HMMs.
    Transitions are more deterministic when the probability mass is more concentrated for transitions from
    all possible states at t-1 to a particular state t. This means that we should be looking at rows in our matrices.

    :param normalized: use variation coefficient if True, else std. dev.
    :return : two arrays of std. deviations
    """
    assert len(nparr1.shape) == 2 & len(nparr2.shape) == 2
    sort_or_no = np.sort if sort else lambda x: x
    var = variation if normalized else np.std
    return sort_or_no(var(nparr1, axis=0)), sort_or_no(var(nparr2, axis=0))


def plot_sparsity_comparison(nparr1, nparr2, sort=True, normalized=True):
    import matplotlib.pyplot as plt

    devs1, devs2 = compare_sparsity(nparr1, nparr2, sort=sort, normalized=normalized)
    plt.plot(list(range(len(devs1))), devs1,
             list(range(len(devs2))), devs2)
    plt.show()


def get_best_clusterids(emission_matrix, i, n, prob_thresh=0.00001):
    """
    For a word i, get n best clusters that exceed prob_thresh.
    prob_thresh can be low because of constraint from "n".

    :param i: row index (word)
    :param n: number of best clusters for a word
    """

    bigger = np.where(emission_matrix[i, :] > prob_thresh)[0].size
    if bigger > n:
        return np.argsort(emission_matrix[i, :])[::-1][:n]
    else:
        return np.argsort(emission_matrix[i, :])[::-1][:bigger]


def posttype_txt(posttypes, vocab_f, threedim, vocab_r):
    """
    produce format as in word embeddings (lxmls.readers.word2vec.py)
    :param posttypes: loaded posttype.npy file
    :param vocab_f: vocabulary of the training text used for obtaining posttype.npy
    """
    w_dict = LabelDictionary(read_vocab(vocab_f))

    if threedim:
        import pickle

        r_dict = pickle.load(open(vocab_r, "rb"))

        rep_iter = ((w_dict.get_label_name(c), rep) for c, rep in enumerate(posttypes))
        return (("{}{}".format(w, r_dict.get_label_name(r)), rep[:, r])
                for w, rep in rep_iter
                for r in range(rep.shape[1]))
    else:
        return ((w_dict.get_label_name(c), rep) for c, rep in enumerate(posttypes))


def plain_posttype_txt(posttype_f, vocab_f, threedim, vocab_r):
    posttypes = np.load(posttype_f)
    m, n = posttypes.shape[0], posttypes.shape[1]
    return m, n, posttype_txt(posttypes, vocab_f, threedim, vocab_r)


def posttype_txt_w2v(posttype_f, vocab_f, output_f):

    m, n, posttypes = plain_posttype_txt(posttype_f, vocab_f)
    write_embed(posttypes, output_f, header="{} {}\n".format(m, n))


def posttype_txt_plain(posttype_f, vocab_f, output_f_v, output_f_rep, threedim=False, vocab_r=None):
    """
    Create two txt files: vocabulary with one word per line, and representation vectors, one per line.

    :param threedim: npy posttypes file contains 3-dimensions, ie extra dim. for syn. fun.
    """
    if threedim and not vocab_r: sys.exit("Missing rel. vocabulary.")
    if vocab_r and not threedim: sys.exit("Use rel. vocabulary?")

    _, _, posttypes = plain_posttype_txt(posttype_f, vocab_f, threedim, vocab_r)
    with open(output_f_v, "w") as out_v, open(output_f_rep, "w") as out_r:
        for w, rep in posttypes:
            if np.isnan(np.sum(rep)):
                continue
            out_v.write("{}\n".format(w))
            out_r.write("{}\n".format(nparr_to_str(rep)))


class EmissionProb:
    """
    Post-training operations on Emission probability matrix (M*N; M vocab size, N n. of states)
    """

    def __init__(self):
        self.e = True

    def get_cluster_by_thresh(self, i, prob_thresh=0.001):
        """
        Ids of those labels (words) that exceed threshold, returned sorted by probability
        :param i: column index (state)
        :param prob_thresh: words below threshold are ignored
        """
        n_ids = np.where(self.ep[:, i] > prob_thresh)[0].size
        # pick relevant, sorted
        return np.argsort(self.ep[:, i])[::-1][:n_ids]

    def get_cluster_by_nmax_prob(self, i, n):
        """
        :param i: column index (state)
        :param n: n words with highest probability
        """
        pass

    def get_label_names(self, indices, x_dict):
        """
        Get labels (words) for indices.

        :param indices: a list of indices
        """

        return [self.get_label_name(x_dict, i) for i in indices]

    def get_relation_names(self, indices, r_dict):
        """
        Get labels (words) for indices.

        :param indices: a list of indices
        """
        return [self.get_label_name(r_dict, i) for i in indices]

    def get_label_name(self, x_dict, i):
        return x_dict.get_label_name(i)

    def get_label_ids(self, x_dict, words):
        """
        Get indices for words.
        :param words: a list of words
        """
        return [self.get_label_id(x_dict, w) for w in words]

    def get_relation_ids(self, words):
        return [self.get_relation_id(w) for w in words]

    def get_label_id(self, x_dict, w):
        return x_dict.get_label_id(w)

    def get_relation_id(self, r_dict, w):
        return self.get_label_id(r_dict, w)

    def get_clusters(self):
        """
        1*N list of lists of  n best words.
        For each cluster id (state) it gets the words if thresh is satisfied.
        """

        N = self.ep.shape[1]
        x_dict = self.data.x_dict
        self.prob_thresh = 0.001
        return [self.get_label_names(self.get_cluster_by_thresh(i, self.prob_thresh), x_dict) for i in range(N)]

    def get_clusters_by_word(self):
        """
        - get best (high prob) clusters for a word
        - with constraints on prob and n of clusters per word
        """
        from collections import defaultdict

        self.prob_thresh = 0.00001
        clusters_per_w = {}
        M = self.ep.shape[0]
        self.n = 5  # max n of clusters per word

        for i in range(M):
            clusterids = get_best_clusterids(self.ep, i, self.n, self.prob_thresh) + 1  # +1 because index start at 0
            clusters_per_w[i] = clusterids

        clusters = defaultdict(list)
        # inverting clusters : get words per cluster
        for i, clusterids in clusters_per_w.items():
            i_name = self.get_label_name(self.data.x_dict, i)  # get word
            if len(clusterids) == 1:  # don't attach word-index if w has only one "sense"/cluster
                clusters[clusterids[0]].append(i_name)
            else:
                for w_i, clusterid in enumerate(clusterids, 1):  # maintain number of "senses"/clusters per word
                    clusters[clusterid].append("{}__{}".format(i_name, w_i))

        return clusters

    def get_clusters_by_word_sorted(self):
        """
         - get best (high prob) clusters for a word such that final clusters include words sorted by prob
         - with constraints on prob and n of clusters per word
        """
        from collections import defaultdict

        self.prob_thresh = 0.00001
        clusters_per_w = {}
        M = self.ep.shape[0]
        self.n = 5  # max n of clusters per word

        for i in range(M):
            clusterids = get_best_clusterids(self.ep, i, self.n, self.prob_thresh)
            clusters_per_w[i] = clusterids

        clusters = defaultdict(list)
        # inverting clusters : get words per cluster
        for i, clusterids in clusters_per_w.items():
            i_name = self.get_label_name(self.data.x_dict, i)  # get word
            for w_i, clusterid in enumerate(clusterids, 1):  # maintain number of "senses"/clusters per word
                #lookup p(i|clusterid)
                p = self.ep[i, clusterid]
                clusters[clusterid].append((p, "{}__{}".format(i_name, w_i)))

        #now sort each entry by p
        clusters_sorted = {clusterid + 1: sorted(v, reverse=True) for clusterid, v in clusters.items()}

        return clusters_sorted

    def hellinger(self, v1, v2):
        #this is inefficient:
        if np.isnan(np.sum(v1)) or np.isnan(np.sum(v2)):
            return 0
        return 1 - norm(np.sqrt(v1) - np.sqrt(v2)) / np.sqrt(2.0)

    def cosine(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def sim_runs_ep(self, i, measure="cosine"):
        if measure == "hellinger":
            sim_run = self.hellinger
        else:
            sim_run = self.cosine
        sims = []
        for k in range(self.ep.shape[0]):
            sim = sim_run(self.ep[i], self.ep[k])
            if sim is not None:
                sims.append(sim)
        return sims

    def sim_runs(self, nparr, i, measure="cosine"):
        assert nparr.ndim == 2
        if measure == "hellinger":
            sim_run = self.hellinger
        else:
            sim_run = self.cosine
        sims = []
        for k in range(nparr.shape[0]):
            sim = sim_run(nparr[i], nparr[k])
            if sim is not None:
                sims.append(sim)
        return sims

    def similarities_query(self, nparr, words, x_dict, top=20, measure="cosine"):
        ids = self.get_label_ids(x_dict, words)
        sim_d = {}
        for i in ids:
            sim_d[i] = np.argsort(self.sim_runs(nparr, i, measure=measure))[-top:][::-1]
        return self.sim_ids_to_words(sim_d, x_dict)

    def similarities(self):
        """
        get top x similar ids of words based on cosine distance between word vectors over states
        """
        cos = np.zeros((self.ep.shape[0], self.ep.shape[0]))
        for i in range(self.ep.shape[0]):
            print(i)
            cos[i, i:] = self.sim_runs_ep(i)

        return self.symmetrize(cos)

    def top_sim(self, cos, top=20):
        cos = cos.argsort()
        return {i: cos[i, -top:] for i in range(cos.shape[0])}

    def symmetrize(self, a):
        """
        fill one matrix side of the diagonal by mirroring
        """
        return a + a.T - np.diag(a.diagonal())

    def sim_ids_to_words(self, sim_d, x_dict):
        return {self.get_label_name(x_dict, k): self.get_label_names(v, x_dict) for k, v in sim_d.items()}

    def write_clusters(self):
        """
        write method for get_clusters.

        """
        with open("{}/clusters".format(self.path), "w") as outfile:
            if self.prob_thresh:
                outfile.write("Words for which p(w|c)>{}\n".format(self.prob_thresh))
            for id_c, c in enumerate(self.clusters, 1):
                for w in c:
                    outfile.write("{}\t{}\n".format(id_c, w))

    def write_clusters_by_word(self):
        """
        write method for get_clusters_by_word.
        """
        with open("{}/clusters".format(self.path), "w") as outfile:
            if self.prob_thresh and self.n:
                outfile.write(
                    "Words for which p(w|c)>{}. Max. {} clusters per word.\n".format(self.prob_thresh, self.n))
            for id_c, c in self.clusters.items():
                for w in c:
                    outfile.write("{}\t{}\n".format(id_c, w))

    def write_clusters_by_word_sorted(self):
        """
        write method for get_clusters_by_word_sorted.
        """
        with open("{}/clusters".format(self.path), "w") as outfile:
            if self.prob_thresh and self.n:
                outfile.write(
                    "Words for which p(w|c)>{}. Max. {} clusters per word.\n".format(self.prob_thresh, self.n))
            for id_c, c in self.clusters.items():
                outfile.write("-------------------------------------------------------------------------------------\n")
                for p, w in c:
                    outfile.write("{}\t{}\n".format(id_c, w))

    def main(self, path):
        """

        :param path: path to dir containing npy and settings files from the experiment
        """
        self.path = path
        self.ep = np.load("{}/ep.npy".format(self.path))
        #get some info from the setttings file
        with open("{}/settings".format(self.path)) as infile:
            data_name = None
            n_sent = None
            for l in infile:
                if l.startswith("Name of the corpus file: "):
                    data_name = l.strip().split(" ")[-1]
                elif l.startswith("Number of sentences: "):
                    n_sent = l.strip().split(" ")[-1]
            if data_name is None:
                print("Not able to retrieve the dataset name.")
            if n_sent is None:
                print("Not able to retrieve the number of sentences.")

        self.data_name = data_name
        self.n_sent = eval(n_sent)
        if "tree" or "_rel_" in path:
            if "_en_" in path:
                self.data = ConllCorpus(self.data_name, howbig=self.n_sent, lemmas=False)
            elif "_nl_" in path:
                self.data = ConllCorpus(self.data_name, howbig=self.n_sent)
        else:
            self.data = TextCorpus(self.data_name, howbig=self.n_sent)

        self.prob_thresh = None
        self.n = None  # max n of clusters per w


def similar_query(posttypes_file, words, vocab_file, top=20, measure="cosine"):
    vocab = LabelDictionary(read_vocab(vocab_file))
    posttypes = np.load(posttypes_file)
    o = EmissionProb()
    return o.similarities_query(posttypes, words, vocab, top=top, measure=measure)


if __name__ == "__main__":
    print(similar_query(
        "posttype.npy",
        ["the"],
        "vocab",
        top=4, measure="hellinger"))