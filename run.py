import argparse
import logging
import sys

import numpy as np

from hmm import prepare_dirname, HMM
from hmrtm import HMRTM
from hmtm import HMTM
from readers.conll07_reader import Conll07Reader
from readers.conll_corpus import ConllCorpus
from readers.text_corpus import TextCorpus
from util.util import line_reader


parser = argparse.ArgumentParser()
parser.add_argument("--a", type=int, default=4, help="a parameter in online learning")
parser.add_argument("--alpha", type=float, default=1, help="alpha parameter in online learning")
parser.add_argument("--append_string", default="",
                    help="string to append to output directory to make it distinguishable")
parser.add_argument("--approx", action='store_true', default=False,
                    help="approximate inference through vector projection")
parser.add_argument("-brown", "--brown_init_path",
                    help="path to file with brown clusters; used for initialization of parameters")
parser.add_argument("-d", "--dataset", help="input corpus")
parser.add_argument("-desired", "--desired_n_states", type=int, help="desired number of states at the end")
parser.add_argument("--full_batch", action="store_true", help="use pure batch EM")
parser.add_argument("--lang", default="en",
                    help="Language, needed only to determine the set of syntactic relations to use.")
parser.add_argument("--lr", action='store_true', default=False,
                    help="induce tree representations using a directional model discriminating between left/right contexts")
parser.add_argument("--max_iter", type=int, help="maximum number of iterations to perform")
parser.add_argument("--minibatch_size", type=int, default=1000, help="size of the minibatch in online learning")
parser.add_argument("--n_proc", type=int, help="number of processors")
parser.add_argument("--noise_amount", type=float, default=0.0001,
                    help="relative amount of noise to add to split parameters")
parser.add_argument("-nw", "--no_writeout", dest="writeout", action='store_false',
                    help="don't write out the matrices and settings")
parser.add_argument("--omit_class_cond", action='store_true', default=False,
                    help="do not condition the class variable on the relation variable in a relational model")
parser.add_argument("--omit_emis_cond", action='store_true', default=False,
                    help="do not condition the emission/word variable on the relation variable in a relational model")
parser.add_argument("-p", "--params", help="path to fixed parameters")
parser.add_argument("--permute", action='store_true', default=True, help="permute minibatch")
parser.add_argument("--rel", action='store_true', default=False, help="induce tree representations using relations")
parser.add_argument("--rel_spec_nl", choices=[
    "mod", "obj1", "punct", "det", "ROOT", "su", "cnj", "mwp", "body", "vc", "predc", "pc", "app", "svp", "ld", "obj2",
    "predm", "se", "obcomp", "me", "sup", "crd", "hdf", "pobj1"
], nargs="+", help="specify the Dutch relation names to keep")
parser.add_argument("--rel_spec_en", choices=[
    "NMOD", "P", "PMOD", "SBJ", "OBJ", "ROOT", "ADV", "DEP", "VC", "NAME", "CONJ", "COORD", "PRD", "IM", "LOC", "OPRD",
    "TMP", "AMOD", "SUB", "APPO", "SUFFIX", "TITLE", "DIR", "POSTHON", "LGS", "PRT", "LOC-PRD", "EXT", "PRP", "MNR",
    "DTV", "PUT", "EXTR", "PRN", "GAP-LGS", "VOC", "LOC-OPRD", "GAP-NMOD", "DEP-GAP", "GAP-OBJ", "GAP-PRD"
], nargs="+", help="specify the English relation names to keep")
parser.add_argument("-s", "--sensitivity", type=float, default=0,
                    help="merging sensitivity. 0 means no merging is done")
parser.add_argument("-start", "--start_n_states", type=int, help="starting number of states")
parser.add_argument("-trained", "--params_trained", action='store_true',
                    help="fixed parameters represent trained parameters")
parser.add_argument("--tree", action='store_true', default=False, help="induce tree representations")
parser.add_argument("-w", "--writeout", dest="writeout", action='store_true',
                    help="write out the matrices and settings")
parser.set_defaults(writeout=True)
args = parser.parse_args()

if "conll" in args.dataset and not (args.tree or args.rel or args.lr):
    sys.exit("--tree possibly missing")

lemmas = False if ((args.tree or args.rel or args.lr) and "bllip" in args.dataset) else True

if args.rel:
    hmm_type = "rel"
elif args.lr:
    hmm_type = "lr"
elif args.tree:
    hmm_type = "tree"
else:
    hmm_type = ""
max_iter = args.max_iter
n_proc = args.n_proc
# set sensitivity to 0 to bypass merging and loss calculation
# online EM
minibatch_size = args.minibatch_size
alpha = args.alpha
a = args.a
permute = args.permute

sensitivity = args.sensitivity
noise_amount = args.noise_amount
append_string = args.append_string

start_n_states = args.start_n_states
# final number of states
desired_n = args.desired_n_states
sm = True  # split-merge procedure

if start_n_states is None:
    # no split-merge
    sm = False
    start_n_states = desired_n

n_sent = 0
if args.tree or args.rel or args.lr:
    reader = Conll07Reader(args.dataset)
    sent = reader.getNext()
    while sent:
        n_sent += 1
        sent = reader.getNext()
else:
    for l in line_reader(args.dataset):
        n_sent += 1

dirname = prepare_dirname(hmm_type=hmm_type, append_string=append_string, lang=args.lang, max_iter=max_iter,
                          N=start_n_states, n_sent=n_sent, alpha=alpha, minibatch_size=minibatch_size)

if args.tree or args.rel or args.lr:
    if args.lang == "en":
        dataset = ConllCorpus(args.dataset, howbig=n_sent, lemmas=lemmas, spec_rels=args.rel_spec_en,
                              dirname=dirname, lr=args.lr)
    elif args.lang == "nl":
        dataset = ConllCorpus(args.dataset, howbig=n_sent, lemmas=lemmas, spec_rels=args.rel_spec_nl,
                              dirname=dirname, lr=args.lr)
    else:
        dataset = ConllCorpus(args.dataset, howbig=n_sent, lemmas=lemmas, spec_rels=None,
                              dirname=dirname, lr=args.lr)
    n_rels = len(dataset.r_dict)
else:
    dataset = TextCorpus(args.dataset, howbig=n_sent)
    dataset.prepare_chains()

n_obs = len(dataset.x_dict)

writeout = args.writeout

if args.rel or args.lr:
    model = HMRTM
elif args.tree:
    model = HMTM
else:
    model = HMM
if args.params is not None:
    params_fixed_path = args.params
    if args.params_trained:
        params_fixed = (np.load("{}ip.npy".format(params_fixed_path)),
                        np.load("{}tp.npy".format(params_fixed_path)),
                        np.load("{}fp.npy".format(params_fixed_path)),
                        np.load("{}ep.npy".format(params_fixed_path)))
    else:
        params_fixed = (np.load("{}ip_init.npy".format(params_fixed_path)),
                        np.load("{}tp_init.npy".format(params_fixed_path)),
                        np.load("{}fp_init.npy".format(params_fixed_path)),
                        np.load("{}ep_init.npy".format(params_fixed_path)))

    if args.rel or args.lr:
        h = model(start_n_states, n_obs, R=n_rels, params=params_fixed, writeout=writeout, approx=args.approx,
                  dirname=dirname, omit_class_cond=args.omit_class_cond, omit_emis_cond=args.omit_emis_cond)
    else:
        h = model(start_n_states, n_obs, params=params_fixed, writeout=writeout, approx=args.approx, dirname=dirname)
    h.params_fixed_path = params_fixed_path
    h.params_fixed_type = args.params_trained
else:
    if args.brown_init_path is not None:
        if args.rel or args.lr:
            h = model(start_n_states, n_obs, R=n_rels, params=args.params, writeout=writeout,
                      brown_init_path=args.brown_init_path, x_dict=dataset.x_dict, approx=args.approx, dirname=dirname,
                      omit_class_cond=args.omit_class_cond, omit_emis_cond=args.omit_emis_cond)
        else:
            h = model(start_n_states, n_obs, params=args.params, writeout=writeout,
                      brown_init_path=args.brown_init_path, x_dict=dataset.x_dict, approx=args.approx, dirname=dirname)
    else:
        if args.rel or args.lr:
            h = model(start_n_states, n_obs, R=n_rels, params=args.params, writeout=writeout, approx=args.approx,
                      dirname=dirname, omit_class_cond=args.omit_class_cond, omit_emis_cond=args.omit_emis_cond)
        else:
            h = model(start_n_states, n_obs, params=args.params, writeout=writeout, approx=args.approx, dirname=dirname)

#EM

if not sm:
    if args.full_batch:
        h.em(dataset, max_iter, hmm_type=hmm_type, append_string=append_string) if n_proc < 2 else h.em_multiprocess(
            dataset, max_iter, n_proc=n_proc, hmm_type=hmm_type, append_string=append_string)
    else:
        # do not use split-merge
        h.online_em(dataset, max_iter, minibatch_size=minibatch_size, alpha=alpha, a=a, permute=permute,
                    hmm_type=hmm_type, append_string=append_string) if n_proc < 2 else h.online_em_multiprocess(dataset,
                                                                                                                max_iter,
                                                                                                                minibatch_size=minibatch_size,
                                                                                                                alpha=alpha,
                                                                                                                a=a,
                                                                                                                permute=permute,
                                                                                                                n_proc=n_proc,
                                                                                                                hmm_type=hmm_type,
                                                                                                                append_string=append_string)
elif sm:
    if args.full_batch:
        h.em_splitmerge(dataset, max_iter, desired_n=desired_n, sensitivity=sensitivity, noise_amount=noise_amount,
                        n_proc=n_proc, logging_level=logging.DEBUG, hmm_type=hmm_type, append_string=append_string)
    else:
        h.em_splitmerge(dataset, max_iter, minibatch_size=minibatch_size, alpha=alpha, a=a, permute=permute,
                        desired_n=desired_n, sensitivity=sensitivity, noise_amount=noise_amount, n_proc=n_proc,
                        logging_level=logging.DEBUG, hmm_type=hmm_type, append_string=append_string)



#write clusters
#from hmm.output.emission_prob import *
#e = EmissionProb(h.dirname)
#clusters = e.get_clusters_by_word_sorted()
#e.write_clusters_by_word_sorted()



