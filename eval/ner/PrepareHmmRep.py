import sys

import numpy as np

from eval.ner.readers.Conll2002NerCorpus import Conll2002NerCorpus, ned_train, ned_dev, ned_test, ned_test_parsed, \
    ned_dev_parsed, ned_train_parsed, ned_dev_parsed_files_path, ned_test_parsed_files_path, ned_train_parsed_files_path
from eval.ner.readers.Conll2003NerCorpus import Conll2003NerCorpus, eng_dev, eng_train, eng_test, muc_test, \
    muc_test_parsed, eng_test_parsed, eng_dev_parsed, eng_train_parsed
from hmm import HMM
from hmrtm import HMRTM
from hmtm import HMTM
from readers.conll_corpus import ConllCorpus
from readers.text_corpus import TextCorpus


def read_params_from_path(path):
    with open("{}/settings".format(path)) as infile:
        lines = infile.readlines()

    n_states = eval(lines[1].rstrip().split()[-1])
    n_obs = eval(lines[2].rstrip().split()[-1])
    n_sent = eval(lines[3].rstrip().split()[-1])
    n_toks = eval(lines[4].rstrip().split()[-1])
    corpus_file = lines[5].rstrip().split()[-1]
    omit_class_cond = None
    omit_emis_cond = None
    for l in lines[5:]:
        try:
            if l.strip().split(":")[0] == "Omit class conditioning":
                omit_class_cond = eval(l.strip().split()[-1])
            elif l.strip().split(":")[0] == "Omit emis conditioning":
                omit_emis_cond = eval(l.strip().split()[-1])
        except IndexError:
            continue

    return n_states, n_obs, n_sent, n_toks, corpus_file, omit_class_cond, omit_emis_cond


class PrepareHmmRep():
    """
    Applying hmm-based representations to the evaluation dataset. This includes decoding.

    """

    def __init__(self, path, lang, decoding=None, use_wordrep_tree=False, use_wordrep_rel=False, eval_spec_rel=False,
                 logger=None, ignore_rel=None, lr=False):

        self.path = path
        self.lang = lang
        self.decoding = decoding
        self.use_wordrep_tree = use_wordrep_tree
        self.use_wordrep_rel = use_wordrep_rel
        self.eval_spec_rel = eval_spec_rel
        if self.decoding == None:
            print("Decoding method not specified.")
            if self.use_wordrep_tree or self.use_wordrep_rel:
                self.decoding = "max-product"
            else:
                self.decoding = "viterbi"
        print("Using default: {}".format(self.decoding))
        self.n_states = None
        self.n_obs = None
        self.n_sent = None
        self.n_toks = None
        self.corpus_file = None
        self.logger = logger
        self.n_states, self.n_obs, self.n_sent, self.n_toks, self.corpus_file, self.omit_class_cond, self.omit_emis_cond = \
            read_params_from_path(self.path)
        if self.logger is not None:
            self.logger.debug("Preparing self.dataset")
        if self.use_wordrep_tree or self.use_wordrep_rel:
            lemmas = False if self.lang == "en" else True
            self.dataset = ConllCorpus("{}".format(self.corpus_file), howbig=self.n_sent, lemmas=lemmas,
                                       eval_spec_rels=self.eval_spec_rel, dirname=self.path, lr=lr)
            self.ignore_rel = self.dataset.r_dict.get_label_id(ignore_rel) if ignore_rel is not None else None
            if decoding == "posterior_cont_type":
                self.dataset.train = self.dataset.prepare_trees_gen()  # generator
        else:
            self.dataset = TextCorpus("{}".format(self.corpus_file), howbig=self.n_sent)
            if decoding == "posterior_cont_type":
                self.dataset.prepare_chains()

        self.ner_corpus = None

        if self.lang == "nl" and not (self.use_wordrep_tree or self.use_wordrep_rel):
            self.train_seq, self.dev_seq, self.test_seq = self.prepare_seqs_nl(self.decoding)
            # self.test_seq = self.prepare_seqs_nl(self.decoding)
        elif self.lang == "nl" and (self.use_wordrep_tree or self.use_wordrep_rel):
            self.train_seq, self.dev_seq, self.test_seq = self.prepare_trees_nl(self.decoding, lr=lr)
            # self.test_seq = self.prepare_trees_nl(self.decoding)
        elif self.lang == "en" and not (self.use_wordrep_tree or self.use_wordrep_rel):
            self.train_seq, self.dev_seq, self.test_seq, self.muc_seq = self.prepare_seqs_en(self.decoding)
        elif self.lang == "en" and (self.use_wordrep_tree or self.use_wordrep_rel):
            self.train_seq, self.dev_seq, self.test_seq, self.muc_seq = self.prepare_trees_en(self.decoding, lr=lr)
        else:
            sys.exit("invalid option in PrepareHmmRep")

        self.dataset = None


    def prepare_seqs_nl(self, decoding="viterbi"):
        params_fixed = (np.load("{}/ip.npy".format(self.path)),
                        np.load("{}/tp.npy".format(self.path)),
                        np.load("{}/fp.npy".format(self.path)),
                        np.load("{}/ep.npy".format(self.path)))

        h = HMM(self.n_states, self.n_obs, params=params_fixed, writeout=False, dirname=self.path)

        self.ner_corpus = Conll2002NerCorpus(self.dataset.x_dict, eval_spec_rel=self.eval_spec_rel, dirname=self.path)
        train_seq = self.ner_corpus.read_sequence_list_conll(ned_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(ned_dev)
        test_seq = self.ner_corpus.read_sequence_list_conll(ned_test)

        decoder = None
        type_decoder = None
        if decoding == "viterbi":
            decoder = h.viterbi_decode_corpus
        elif decoding == "max_emission":
            decoder = h.max_emission_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            type_decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined, using Viterbi.")
            decoder = h.viterbi_decode_corpus

        print("Decoding word representations on train.")
        type_decoder(train_seq, self.dataset, self.logger) if type_decoder is not None else decoder(train_seq)
        print("Decoding word representations on dev.")
        type_decoder(dev_seq, self.dataset, self.logger) if type_decoder is not None else decoder(dev_seq)
        print("Decoding word representations on test.")
        type_decoder(test_seq, self.dataset, self.logger) if type_decoder is not None else decoder(test_seq)

        return train_seq, dev_seq, test_seq

    def prepare_seqs_en(self, decoding="viterbi"):
        params_fixed = (np.load("{}/ip.npy".format(self.path)),
                        np.load("{}/tp.npy".format(self.path)),
                        np.load("{}/fp.npy".format(self.path)),
                        np.load("{}/ep.npy".format(self.path)))

        h = HMM(self.n_states, self.n_obs, params=params_fixed, writeout=False, dirname=self.path)

        self.ner_corpus = Conll2003NerCorpus(self.dataset.x_dict)

        train_seq = self.ner_corpus.read_sequence_list_conll(eng_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(eng_dev)
        test_seq = self.ner_corpus.read_sequence_list_conll(eng_test)
        muc_seq = self.ner_corpus.read_sequence_list_conll(muc_test)

        decoder = None
        type_decoder = None
        if decoding == "viterbi":
            decoder = h.viterbi_decode_corpus
        elif decoding == "max_emission":
            decoder = h.max_emission_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            type_decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined correctly, using Viterbi.")
            decoder = h.viterbi_decode_corpus

        print("Decoding word representations on train.")
        type_decoder(train_seq, self.dataset, self.logger) if type_decoder is not None else decoder(train_seq)
        print("Decoding word representations on dev.")
        type_decoder(dev_seq, self.dataset, self.logger) if type_decoder is not None else decoder(dev_seq)
        print("Decoding word representations on test.")
        type_decoder(test_seq, self.dataset, self.logger) if type_decoder is not None else decoder(test_seq)
        print("Decoding word representations on MUC.")
        type_decoder(muc_seq, self.dataset, self.logger) if type_decoder is not None else decoder(muc_seq)

        return train_seq, dev_seq, test_seq, muc_seq

    def prepare_trees_nl(self, decoding="max-product", lr=False):
        params_fixed = (np.load("{}ip.npy".format(self.path)),
                        np.load("{}tp.npy".format(self.path)),
                        np.load("{}fp.npy".format(self.path)),
                        np.load("{}ep.npy".format(self.path)))

        if self.use_wordrep_rel:
            h = HMRTM(self.n_states, self.n_obs, R=len(self.dataset.r_dict), params=params_fixed, writeout=False,
                      dirname=self.path, omit_class_cond=self.omit_class_cond, omit_emis_cond=self.omit_emis_cond)
        else:
            h = HMTM(self.n_states, self.n_obs, params=params_fixed, writeout=False, dirname=self.path)
        # h.dirname = self.path
        self.logger.debug("Creating self.ner_corpus")
        self.ner_corpus = Conll2002NerCorpus(self.dataset.x_dict, eval_spec_rel=self.eval_spec_rel, dirname=self.path,
                                             lr=lr, use_wordrep_tree=True)

        self.logger.debug("Reading ner data from self.ner_corpus")
        train_seq = self.ner_corpus.read_sequence_list_conll(ned_train, ned_train_parsed, ned_train_parsed_files_path)
        dev_seq = self.ner_corpus.read_sequence_list_conll(ned_dev, ned_dev_parsed, ned_dev_parsed_files_path)
        test_seq = self.ner_corpus.read_sequence_list_conll(ned_test, ned_test_parsed, ned_test_parsed_files_path)

        decoder = None
        type_decoder = None
        if decoding == "max-product":
            decoder = h.max_product_decode_corpus
        #elif decoding == "max_emission":
        #    decoder = h.max_emission_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            type_decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined, using Max-product message passing.")
            decoder = h.max_product_decode_corpus

        self.logger.debug("Decoding.")
        print("Decoding word representations on train.")
        type_decoder(train_seq, self.dataset, self.logger) if type_decoder is not None else decoder(train_seq,
                                                                                                    self.ignore_rel)
        print("Decoding word representations on dev.")
        type_decoder(dev_seq, self.dataset, self.logger) if type_decoder is not None else decoder(dev_seq,
                                                                                                  self.ignore_rel)
        print("Decoding word representations on test.")
        type_decoder(test_seq, self.dataset, self.logger) if type_decoder is not None else decoder(test_seq,
                                                                                                   self.ignore_rel)

        return train_seq, dev_seq, test_seq

    def prepare_trees_en(self, decoding="max-product", lr=False):
        params_fixed = (np.load("{}ip.npy".format(self.path)),
                        np.load("{}tp.npy".format(self.path)),
                        np.load("{}fp.npy".format(self.path)),
                        np.load("{}ep.npy".format(self.path)))

        if self.use_wordrep_rel:
            h = HMRTM(self.n_states, self.n_obs, R=len(self.dataset.r_dict), params=params_fixed, writeout=False,
                      dirname=self.path, omit_class_cond=self.omit_class_cond, omit_emis_cond=self.omit_emis_cond)
        else:
            h = HMTM(self.n_states, self.n_obs, params=params_fixed, writeout=False, dirname=self.path)

        self.logger.debug("Reading ner data from self.ner_corpus")
        self.ner_corpus = Conll2003NerCorpus(self.dataset.x_dict, eval_spec_rel=self.eval_spec_rel, dirname=self.path,
                                             lr=lr)

        train_seq = self.ner_corpus.read_sequence_list_conll(eng_train, eng_train_parsed)
        dev_seq = self.ner_corpus.read_sequence_list_conll(eng_dev, eng_dev_parsed)
        test_seq = self.ner_corpus.read_sequence_list_conll(eng_test, eng_test_parsed)
        muc_seq = self.ner_corpus.read_sequence_list_conll(muc_test, muc_test_parsed)

        # return train_seq, dev_seq, test_seq
        decoder = None
        type_decoder = None
        if decoding == "max-product":
            decoder = h.max_product_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            type_decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined, using Max-product message passing.")
            decoder = h.max_product_decode_corpus

        print("Decoding word representations on train.")
        type_decoder(train_seq, self.dataset, self.logger) if type_decoder is not None else decoder(train_seq,
                                                                                                    self.ignore_rel)
        print("Decoding word representations on dev.")
        type_decoder(dev_seq, self.dataset, self.logger) if type_decoder is not None else decoder(dev_seq,
                                                                                                  self.ignore_rel)
        print("Decoding word representations on test.")
        type_decoder(test_seq, self.dataset, self.logger) if type_decoder is not None else decoder(test_seq,
                                                                                                   self.ignore_rel)
        print("Decoding word representations on MUC.")
        type_decoder(muc_seq, self.dataset, self.logger) if type_decoder is not None else decoder(muc_seq,
                                                                                                  self.ignore_rel)

        return train_seq, dev_seq, test_seq, muc_seq
        #return test_seq


class PrepareHmmRepDbg():
    def __init__(self, path, lang, decoding=None, use_wordrep_tree=False):

        self.path = path
        self.lang = lang
        self.decoding = decoding
        self.use_wordrep_tree = use_wordrep_tree
        if self.decoding == None:
            print("Decoding method not specified.")
            if self.use_wordrep_tree:
                self.decoding = "max-product"
            else:
                self.decoding = "viterbi"
        print("Using default: {}".format(self.decoding))
        self.n_states = None
        self.n_obs = None
        self.n_sent = None
        self.n_toks = None
        self.corpus_file = None

        self.n_states, self.n_obs, self.n_sent, self.n_toks, self.corpus_file = \
            self.read_params_from_path()

        if self.use_wordrep_tree:
            if self.lang == "en":
                self.dataset = ConllCorpus("{}".format(self.corpus_file), howbig=self.n_sent, lemmas=False)
            elif self.lang == "nl":
                self.dataset = ConllCorpus("{}".format(self.corpus_file), howbig=self.n_sent)
        else:
            self.dataset = TextCorpus("{}".format(self.corpus_file), howbig=self.n_sent)
        self.ner_corpus = None

        if self.lang == "nl" and not self.use_wordrep_tree:
            self.dev_seq, self.test_seq = self.prepare_seqs_nl_dbg(self.decoding)
            # self.test_seq = self.prepare_seqs_nl(self.decoding)
        elif self.lang == "nl" and self.use_wordrep_tree:
            self.train_seq, self.dev_seq, self.test_seq = self.prepare_trees_nl(self.decoding)
            # self.test_seq = self.prepare_trees_nl(self.decoding)
        elif self.lang == "en" and not self.use_wordrep_tree:
            self.dev_seq = self.prepare_seqs_en_dbg(self.decoding)

        elif self.lang == "en" and self.use_wordrep_tree:
            self.dev_seq = self.prepare_trees_en_dbg(self.decoding)


    def read_params_from_path(self):
        with open("{}/settings".format(self.path)) as infile:
            lines = infile.readlines()

        n_states = eval(lines[1].rstrip().split()[-1])
        n_obs = eval(lines[2].rstrip().split()[-1])
        n_sent = eval(lines[3].rstrip().split()[-1])
        n_toks = eval(lines[4].rstrip().split()[-1])
        corpus_file = lines[5].rstrip().split()[-1]

        return n_states, n_obs, n_sent, n_toks, corpus_file

    def prepare_seqs_nl_dbg(self, decoding="viterbi"):
        params_fixed = (np.load("{}ip.npy".format(self.path)),
                        np.load("{}tp.npy".format(self.path)),
                        np.load("{}fp.npy".format(self.path)),
                        np.load("{}ep.npy".format(self.path)))

        h = HMM(self.n_states, self.n_obs, params=params_fixed, writeout=False)
        h.dirname = self.path
        self.ner_corpus = Conll2002NerCorpus(self.dataset.x_dict)

        # train_seq = self.ner_corpus.read_sequence_list_conll(ned_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(ned_dev)
        test_seq = self.ner_corpus.read_sequence_list_conll(ned_test)

        if decoding == "viterbi":
            decoder = h.viterbi_decode_corpus
        elif decoding == "max_emission":
            decoder = h.max_emission_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined, using Viterbi.")
            decoder = h.viterbi_decode_corpus

        #print("Decoding word representations on train.")
        #decoder(train_seq)
        print("Decoding word representations on dev.")
        decoder(dev_seq)
        #print("Decoding word representations on test.")
        #decoder(test_seq)

        #return train_seq, dev_seq, test_seq
        #return dev_seq, test_seq

    def prepare_seqs_en_dbg(self, decoding="viterbi"):
        params_fixed = (np.load("{}ip.npy".format(self.path)),
                        np.load("{}tp.npy".format(self.path)),
                        np.load("{}fp.npy".format(self.path)),
                        np.load("{}ep.npy".format(self.path)))

        h = HMM(self.n_states, self.n_obs, params=params_fixed, writeout=False)
        h.dirname = self.path
        self.ner_corpus = Conll2003NerCorpus(self.dataset.x_dict)

        # train_seq = self.ner_corpus.read_sequence_list_conll(eng_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(eng_dev)
        #test_seq = self.ner_corpus.read_sequence_list_conll(eng_test)
        #muc_seq = self.ner_corpus.read_sequence_list_conll(muc_test)


        if decoding == "viterbi":
            decoder = h.viterbi_decode_corpus
        elif decoding == "max_emission":
            decoder = h.max_emission_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined correctly, using Viterbi.")
            decoder = h.viterbi_decode_corpus


        #print("Decoding word representations on train.")
        #decoder(train_seq)
        print("Decoding word representations on dev.")
        decoder(dev_seq, self.dataset)

        #print("Decoding word representations on test.")
        #decoder(test_seq)
        #print("Decoding word representations on MUC.")
        #decoder(muc_seq)

        #return train_seq, dev_seq, test_seq, muc_seq
        return dev_seq

    def prepare_trees_en_dbg(self, decoding="max-product"):
        params_fixed = (np.load("{}ip.npy".format(self.path)),
                        np.load("{}tp.npy".format(self.path)),
                        np.load("{}fp.npy".format(self.path)),
                        np.load("{}ep.npy".format(self.path)))

        h = HMTM(self.n_states, self.n_obs, params=params_fixed, writeout=False)
        h.dirname = self.path
        self.ner_corpus = Conll2003NerCorpus(self.dataset.x_dict)


        # train_seq = self.ner_corpus.read_sequence_list_conll(eng_train, eng_train_parsed)
        dev_seq = self.ner_corpus.read_sequence_list_conll(eng_dev, eng_dev_parsed)
        #    test_seq = self.ner_corpus.read_sequence_list_conll(eng_test, eng_test_parsed)
        #    muc_seq = self.ner_corpus.read_sequence_list_conll(muc_test, muc_test_parsed)

        #return train_seq, dev_seq, test_seq
        decoder = None
        type_decoder = None
        if decoding == "max-product":
            decoder = h.max_product_decode_corpus
        elif decoding == "posterior":
            decoder = h.posterior_decode_corpus
        elif decoding == "posterior_cont":
            decoder = h.posterior_cont_decode_corpus
        elif decoding == "posterior_cont_type":
            type_decoder = h.posterior_cont_type_decode_corpus
        else:
            print("Decoder not defined, using Max-product message passing.")
            decoder = h.max_product_decode_corpus

        print("Decoding word representations on dev.")
        type_decoder(dev_seq, self.dataset) if type_decoder is not None else decoder(dev_seq)

        return dev_seq

