import argparse

import numpy as np

from eval.ner.PrepareHmmRep import read_params_from_path
from hmrtm import HMRTM
from readers.conll_corpus import ConllCorpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rep", "--rep_path", help="directory containing (hmm) word representations files")
    parser.add_argument("--use_lemmas", action='store_true', default=False, help="")
    args = parser.parse_args()

    path = args.rep_path
    posttype_f = "{}posttype_cumul.npy".format(path)
    n_states, n_obs, n_sent, n_toks, corpus_file, omit_class_cond, omit_emis_cond = read_params_from_path(path)
    lemmas = args.use_lemmas
    eval_spec_rel = True
    lr = False

    params_fixed = (np.load("{}ip.npy".format(path)),
                    np.load("{}tp.npy".format(path)),
                    np.load("{}fp.npy".format(path)),
                    np.load("{}ep.npy".format(path)))

    dataset = ConllCorpus("{}".format(corpus_file), howbig=n_sent, lemmas=lemmas, eval_spec_rels=eval_spec_rel,
                          dirname=path, lr=lr)
    dataset.train = dataset.prepare_trees_gen()  # generator
    h = HMRTM(n_states, n_obs, R=len(dataset.r_dict), params=params_fixed, writeout=False, dirname=path,
              omit_class_cond=omit_class_cond, omit_emis_cond=omit_emis_cond)

    h.obtain_posttypes_cumul(posttype_f, dataset, n_types=h.M, logger=None)