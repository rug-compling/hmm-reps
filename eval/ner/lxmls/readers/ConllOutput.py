from sequences.label_dictionary import LabelDictionary
from sequences.sequence_list_label import SequenceListLabel


class ConllOutput():
    def __init__(self, wordrep_dict=None):
        self.word_dict = LabelDictionary()
        self.tag_dict = LabelDictionary()
        self.sequence_list = None


def read_sequence_list_conll(infile, word_dict, tag_dict, max_sent_len=100000, max_nr_sent=100000):
    instance_list, word_dict, tag_dict = read_output_instances(infile, word_dict, tag_dict, max_sent_len, max_nr_sent)

    seq_list = SequenceListLabel(word_dict, tag_dict, tag_dict)
    for sent_x, sent_gold, sent_predict in instance_list:
        seq_list.add_sequence(sent_x, sent_gold, sent_predict)
    return seq_list, word_dict, tag_dict


def read_output_instances(file, word_dict, tag_dict, max_sent_len, max_nr_sent):
    contents = open(file, encoding="ascii")
    nr_sent = 0
    instances = []
    ex_x = []
    ex_gold = []
    ex_predict = []

    for line in contents:
        if "-DOCSTART" in line:
            continue
        toks = line.strip().split(" ")
        if len(toks) < 3:
            if max_sent_len > len(ex_x) > 0:  # len(ex_x) > 1 # escape one-word sentences
                nr_sent += 1
                instances.append([ex_x, ex_gold, ex_predict])
            if nr_sent >= max_nr_sent:
                break
            ex_x = []
            ex_gold = []
            ex_predict = []
        else:

            word = toks[0]
            gold = toks[1]
            predicted = toks[2]
            if word not in word_dict:
                word_dict.add(word)
            if gold not in tag_dict:
                tag_dict.add(gold)
            if predicted not in tag_dict:
                tag_dict.add(predicted)
            ex_x.append(word)
            ex_gold.append(gold)
            ex_predict.append(predicted)

    return instances, word_dict, tag_dict


def write_conll_instances(gold, predictions, file, sep=" ", is_muc=False):
    """
    Create dataset with appended predictions as the last column.

    :param is_muc: MUC-7 dataset, remove MISC labels from predictions
    """
    assert len(gold) == len(predictions)
    contents = open(file, "w", encoding="ascii")
    for gold_seq, pred_seq in zip(gold.seq_list, predictions):
        for x, y, y_hat in zip(gold_seq.x, gold_seq.y, pred_seq.y):
            y_hat_label = pred_seq.sequence_list.y_dict.get_label_name(y_hat)
            if is_muc:
                y_hat_label = "O" if "MISC" in y_hat_label else y_hat_label
            contents.write("{}{sep}{}{sep}{}\n".format(gold_seq.sequence_list.x_dict.get_label_name(x),
                                                       gold_seq.sequence_list.y_dict.get_label_name(y),
                                                       y_hat_label,
                                                       sep=sep))
        contents.write("\n")


# Dumps a corpus into a file
def save_corpus(self, dir):
    raise NotImplementedError()


# Loads a corpus from a file
def load_corpus(self, dir):
    raise NotImplementedError()

