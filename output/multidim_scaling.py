import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS

from output.emission_prob import plain_posttype_txt
from sequences.label_dictionary import LabelDictionary
from readers.vocab import read_vocab
from util.util import nparr_to_str


def get_w_indices(targets, vocab):
    if not targets:
        return {}
    w_dict = LabelDictionary(read_vocab(vocab))
    return {w_dict.get_label_id(t) for t in targets if t in w_dict}


def get_w_reps(idx, w_reps, vocab):
    ws = []
    reps = []
    if not idx:
        return ws, reps

    w_dict = LabelDictionary(read_vocab(vocab))
    for w, rep in w_reps:
        if w_dict.get_label_id(w) in idx:
            assert not np.isnan(np.sum(rep))
            ws.append(w)
            reps.append(rep)

    return ws, reps


def get_twodim_reps(reps, seed, distance=euclidean_distances):
    reps = reps.astype(np.float64)
    similarities = distance(reps)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed)
    return mds.fit(similarities).embedding_


def plot(scaled, ws):
    fig, ax = plt.subplots()
    ax.scatter(scaled[:, 0], scaled[:, 1])
    for i, w in enumerate(ws):
        ax.annotate(w, (scaled[i, 0], scaled[i, 1]))
    return fig


def write_fig_data(reps, ws, outfile):
    with open(outfile, "w") as out:
        for w, arr in zip(ws, reps):
            out.write("{} {}\n".format(w, nparr_to_str(arr)))


def expand_w_reps(rep_file, ws, reps):
    if rep_file is not None:
        with open(rep_file) as infile:
            for l in infile:
                w, rep = l.strip().split(" ", 1)
                num_rep = np.array(rep.split()).astype("f")
                assert not np.isnan(np.sum(num_rep))
                ws.append(w)
                reps.append(num_rep)
    return ws, np.array(reps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-vocab", help="vocabulary file for the (hmm) word representations")
    parser.add_argument("-posttypes", help="npy file containing posterior types")
    parser.add_argument("-targets", nargs="+", help="target words to scale down")
    parser.add_argument("-incontext_file",
                        help="incontext representations for words (optional). These will be added to posterior types")
    parser.add_argument("-outfile", help="file path to write the plot to")
    args = parser.parse_args()
    if args.targets is None:
        targets = set()
        print("No targets specified, using vectors from incontext_file")
    else:
        targets = set(args.targets)
    outfile = os.path.splitext(args.outfile)[0] if args.outfile.endswith(".pdf") else args.outfile

    m, n, w_reps = plain_posttype_txt(posttype_f=args.posttypes, vocab_f=args.vocab, threedim=False, vocab_r=None)

    idx = get_w_indices(targets, args.vocab)

    ws, reps = expand_w_reps(args.incontext_file, *get_w_reps(idx, w_reps, args.vocab))

    scaled = get_twodim_reps(reps, seed=np.random.RandomState(seed=3))  # a m*2-dim np array

    assert len(ws) == scaled.shape[0]
    fig = plot(scaled, ws)

    fig.savefig("{}.pdf".format(outfile))
    write_fig_data(scaled, ws, outfile)

