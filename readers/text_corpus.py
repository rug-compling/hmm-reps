from readers.vocab import read_vocab
from sequences.label_dictionary import LabelDictionary
from sequences.sequence_list import SequenceList

__author__ = 'sim'


class TextCorpus:
    def __init__(self, corpus_file, minfreq=0, howbig=10000):
        """

        :param minfreq: minimum frequency of a word in order to be taken into account
        :param howbig: number of sentences to take into account
        """
        # self.corpus_file = "/home/p262594/Datasets/SONAR/SONAR_random4000000.roots.1M"
        #self.corpus_file = "/home/p262594/Datasets/SONAR/SONAR_random4000000.roots.1M.norm.unk1"
        #self.corpus_file = "/home/p262594/Datasets/SONAR/{}".format(corpus_file)
        #self.corpus_file = "/home/p262594/Datasets/PTB_conll/{}".format(corpus_file)
        #self.corpus_file = "/home/p262594/Datasets/bllip_87_89_wsj/{}".format(corpus_file)
        self.corpus_file = corpus_file
        self.vocab_file = "{}.vocab{}".format(self.corpus_file, howbig)  #file of form: w\tf\n

        self.minfreq = minfreq
        self.howbig = howbig
        try:
            self.x_dict = LabelDictionary(read_vocab(self.vocab_file, self.minfreq))
        except IOError:
            self.prepare_vocab_dict()
            self.x_dict = LabelDictionary(read_vocab(self.vocab_file, self.minfreq))

        print("LabelDictionary created.")

    def prepare_chains(self):

        # training sequences
        self.train = SequenceList(self.x_dict)
        print("Creating training from corpus.")
        with open(self.corpus_file) as IN:
            for c, l in enumerate(IN, 1):
                if c > self.howbig:
                    break
                self.train.add_sequence([w for w in l.strip().split(" ")])

    def prepare_vocab_dict(self):
        from collections import defaultdict

        vocab_dict = defaultdict(int)

        with open(self.corpus_file) as IN, open(self.vocab_file, "w") as OUT:
            for c, l in enumerate(IN, 1):
                if c > self.howbig:
                    break
                for w in l.strip().split(" "):
                    vocab_dict[w] += 1
            for w, f in vocab_dict.items():
                OUT.write("{}\t{}\n".format(w, f))
        print("Vocabulary file prepared.")