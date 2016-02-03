# from pympler.classtracker import *
from sequences.label_dictionary import LabelDictionary
from sequences.sequence_list import SequenceList


""" Example sequence for debugging and testing purposes. """


class ExampleSequence:
    def __init__(self):
        #observation vocabulary
        self.x_dict = LabelDictionary(["walk", "shop", "clean", "tennis"])

        #training sequences
        train_seqs = SequenceList(self.x_dict)
        train_seqs.add_sequence(["walk", "walk", "shop", "clean"])
        train_seqs.add_sequence(["walk", "walk", "shop", "clean"])
        train_seqs.add_sequence(["walk", "shop", "shop", "clean"])

        self.train = train_seqs