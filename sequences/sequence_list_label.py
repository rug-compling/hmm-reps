import sys

from sequences.sequence_label import SequenceLabel
from sequences.sequence_list import SequenceList


__author__ = 'sim'


class SequenceListLabel(SequenceList):
    def __init__(self, x_dict, y_dict, wordrep_dict=None):
        super().__init__(x_dict)
        self.y_dict = y_dict
        self.wordrep_dict = wordrep_dict

    def add_sequence(self, x, y, z=None, t=None):
        """Add a sequence to the list, where x is the sequence of
        observations
        :param z: x as used for wordrep (e.g. with *unk*)
        :param t: parsed x, normalized, as used for tree-based wordrep
        """
        num_seqs = len(self.seq_list)
        x_ids = [self.x_dict.get_label_id(name) for name in x]
        y_ids = [self.y_dict.get_label_id(name) for name in y]

        if z is not None and t is not None:
            sys.exit("Both chain and tree rep provided.")

        if z is not None:
            z_ids = [self.wordrep_dict.get_label_id(name) for name in z]
            self.seq_list.append(SequenceLabel(self, x_ids, y_ids, num_seqs, z_ids))
        elif t is not None:
            self.seq_list.append(SequenceLabel(self, x_ids, y_ids, num_seqs, None, None, t))
        else:
            self.seq_list.append(SequenceLabel(self, x_ids, y_ids, num_seqs))

    def save(self, file):
        seq_fn = open(file, "w")
        for seq in self.seq_list:
            txt = ""
            for pos, x in enumerate(seq.x):
                txt += "{}:{}\t".format(x, seq.y[pos])
            seq_fn.write(txt.strip() + "\n")
        seq_fn.close()

    def load(self, file):
        """ load an indexed sequence list """
        seq_fn = open(file, "r")
        seq_list = []

        for line in seq_fn:
            seq_x = [int(x) for x in line.strip().split("\t")]
            seq_y = [int(y) for y in line.strip().split("\t")]
            self.add_sequence(seq_x, seq_y)
        seq_fn.close()