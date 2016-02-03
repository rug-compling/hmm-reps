import numpy as np
from collections import defaultdict
from util.util import line_reader, nparr_to_str


def prepare_cluster_map(brown_cluster_file):
    """
    Build a word-to-clusterid map.
    """
    mapping = {}
    for l in line_reader(brown_cluster_file):
        c_id, w, fq = l.strip().split("\t")
        mapping[w] = c_id  # keep string cluster ids

    return mapping

def prepare_cluster_to_word_map(brown_cluster_file):
    """
    Build a clusterid-to-word map.
    """
    mapping = defaultdict(set)
    for l in line_reader(brown_cluster_file):
        c_id, w, fq = l.strip().split("\t")
        mapping[c_id].add(w)  # keep string cluster ids

    return mapping

def writeout_cluster_to_word_map(mapping, output_f_v, output_f_rep, replace_ids=True, one_hot=False):
    with open(output_f_v, "w") as out_word, open(output_f_rep, "w") as out_rep:
        if replace_ids:
            new_c_id = 0
            for _, w_set in mapping.items():
                new_c_id += 1
                for w in w_set:
                    out_word.write(w+"\n")
                    out_rep.write("{}\n".format(new_c_id))
        elif one_hot:
            for c, (_, w_set) in enumerate(mapping.items()):
                for w in w_set:
                    out_word.write(w+"\n")
                    one_hot = np.zeros(len(mapping), 'int')
                    one_hot[c] = 1
                    out_rep.write("{}\n".format(nparr_to_str(one_hot)))
