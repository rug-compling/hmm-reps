import argparse

import numpy as np

from eval.ner.PrepareHmmRep import read_params_from_path
from hmrtm import HMRTM
from hmtm import HMTM
from readers.conll_corpus import ConllCorpus
from util.util import nparr_to_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rep", "--rep_path", help="directory containing (hmm) word representations files")
    parser.add_argument("-infile", help="file with sentences for decoding. Conll format for parsed sentences.")
    parser.add_argument("-outfile", help="file to write posteriors to")
    parser.add_argument("--use_lemmas", action='store_true', default=False, help="")
    parser.add_argument("--synfunc", action='store_true', default=False,
                        help="Word representations model is sensitive to syntactic functions. Use flag when using model with names like \"hmm_en_rel_...\".")
    args = parser.parse_args()
    path = args.rep_path
    infile = args.infile

    # obtain model parameters
    n_states, n_obs, _, _, _, omit_class_cond, omit_emis_cond = read_params_from_path(path)
    lemmas = args.use_lemmas
    eval_spec_rel = args.synfunc
    lr = False

    # load model
    params_fixed = (np.load("{}ip.npy".format(path)),
                    np.load("{}tp.npy".format(path)),
                    np.load("{}fp.npy".format(path)),
                    np.load("{}ep.npy".format(path)))


    # prepare sents for decoding
    sents = ConllCorpus(infile, howbig=1000000, lemmas=lemmas, eval_spec_rels=eval_spec_rel, dirname=path, lr=lr)
    sents.prepare_trees()

    h = HMRTM(n_states, n_obs, R=len(sents.r_dict), params=params_fixed, writeout=False, dirname=path,
              omit_class_cond=omit_class_cond, omit_emis_cond=omit_emis_cond) if eval_spec_rel else \
        HMTM(n_states, n_obs, params=params_fixed, writeout=False, dirname=path)

    with open(args.outfile, "w") as out:
        for tree in sents.train:
            # obtain posteriors for all nodes
            node_to_rep = h.posterior_decode(tree, cont=True)
            # get words
            for node in tree.get_nonroots():
                out.write(
                    "{} {}\n".format(sents.x_dict.get_label_name(node.name), nparr_to_str(node_to_rep[node.index])))
            out.write("\n")
