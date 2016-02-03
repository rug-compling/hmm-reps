from sequences.label_dictionary import LabelDictionary
from sequences.sequence_list import SequenceList

data_dir = ""

ned_train = "{}ned.train".format(data_dir)
ned_dev = "{}ned.testa".format(data_dir)
ned_test = "{}ned.testb".format(data_dir)


class Conll2002NerCorpusWordrep:
    """
    Reads text to which we want to apply a wordrep such as hmm.
    The object is assigned externally the x_dict used for training the wordrep.
    - class definition is like Conll2002NerCorpus, but no tag/y-related stuff
    - no update of the word_dict; every word not in it (from x_dict), gets *unk* id needed for successful decoding
    """

    def __init__(self):
        self.word_dict = LabelDictionary()
        self.sequence_list = SequenceList(self.word_dict)

    def read_sequence_list_conll(self, train_file, max_sent_len=100000, max_nr_sent=100000):
        """ Read a conll2002 or conll2003 file into a sequence list."""
        instance_list = self.read_conll_instances(train_file, max_sent_len, max_nr_sent)
        seq_list = SequenceList(self.word_dict)  # for indices
        for sent_x in instance_list:
            seq_list.add_sequence(sent_x)
        return seq_list

    def read_conll_instances(self, file, max_sent_len, max_nr_sent):
        contents = open(file, encoding="iso-8859-1")

        nr_sent = 0
        instances = []
        ex_x = []
        nr_types = len(self.word_dict)

        for line in contents:
            if line.startswith("-DOCSTART"):
                continue
            toks = line.split()
            if len(toks) < 3:
                if (len(ex_x) < max_sent_len and len(ex_x) > 0):  # len(ex_x) > 1 # escape one-word sentences
                    nr_sent += 1
                    instances.append(ex_x)
                if (nr_sent >= max_nr_sent):
                    break
                ex_x = []
            else:
                word = toks[0]
                if word not in self.word_dict:
                    ex_x.append("*unk*")
                else:
                    ex_x.append(word)

        return instances