from collections import defaultdict
import argparse
import logging

from eval.ner.lxmls.readers.ConllOutput import read_sequence_list_conll, ConllOutput


def iter_dataset(dataset):
    for seq in dataset.sequence_list.seq_list:
        for word, gold, predict in zip(seq.x, seq.y, seq.z):
            yield (word, gold, predict)


def setup_logging(filename, level):
    """
    :param level: logging.INFO or logging.DEBUG, etc.
    """
    logging.basicConfig(filename=filename, level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # consoleHandler = logging.StreamHandler(sys.stdout)
    return logging.getLogger(__name__)


def load_data(filename):
    dataset = ConllOutput()
    dataset.sequence_list, dataset.word_dict, dataset.tag_dict = read_sequence_list_conll(filename,
                                                                                          dataset.word_dict,
                                                                                          dataset.tag_dict)

    return dataset


def results(dataset, out_tag):
    """
    Identification-only (label-irrelevant)
    :param out_tag: the label to treat as "outside" in NER; ignored in evaluation
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for word, gold, predict in iter_dataset(dataset):
        if gold == out_tag:
            if predict == out_tag:
                tn += 1
            else:
                fp += 1
        else:
            if predict == out_tag:
                fn += 1
            else:
                tp += 1

    return tp, fp, tn, fn


def precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def recall(tp, fn):
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def fscore(precision, recall):
    if (precision + recall) == 0:
        return 0
    return 2 * ((precision * recall) / (precision + recall))


def results_type(dataset, out_tag):
    """
    :param out_tag: the label to treat as "outside" in NER; ignored in evaluation
    """
    # counts = {}  # {w : {correct: 0, incorrect: 0}}
    correct = defaultdict(int)
    incorrect = defaultdict(int)

    for word, gold, predict in iter_dataset(dataset):
        if gold != out_tag:
            if gold == predict:
                correct[word] += 1
            else:
                incorrect[word] += 1

    return correct, incorrect


def results_type_fscore(dataset, out_tag):
    """
    :param out_tag: the label to treat as "outside" in NER; ignored in evaluation
    """

    class Counts():
        def __init__(self):
            self.tp = 0
            self.fp = 0
            self.tn = 0
            self.fn = 0

    counts_type = defaultdict(Counts)

    for word, gold, predict in iter_dataset(dataset):
        if gold == out_tag:
            if predict == out_tag:
                counts_type[word].tn += 1
            else:
                counts_type[word].fp += 1
        else:
            if predict == out_tag:
                counts_type[word].fn += 1
            else:
                counts_type[word].tp += 1

    return counts_type


def fscore_type(counts):
    fscores = {}
    for type in counts.keys():
        fscores[type] = fscore(precision(counts[type].tp, counts[type].fp), recall(counts[type].tp, counts[type].fn))
    return fscores


def accuracy(correct_type, incorrect_type):
    def _accuracy(n_pos, n_neg):
        return n_pos / (n_pos + n_neg)

    types = correct_type.keys() | incorrect_type.keys()
    return {type: _accuracy(correct_type[type], incorrect_type[type]) for type in types}


def score_overall(scores_type, skip_zero=False):
    total_score = 0
    if skip_zero:
        zeros = 0
    for score in scores_type.values():
        if skip_zero and score == 0:
            zeros += 1
            continue
        total_score += score
    return (total_score / len(scores_type)) if not skip_zero else  (total_score / (len(scores_type) - zeros))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input ner conll file")
    parser.add_argument("-o", help="path name for the output file", default=None)
    args = parser.parse_args()
    outfile = "{}.result".format(args.i) if args.o is None else args.o

    # get logger
    logger = setup_logging("{}.log".format(outfile), logging.INFO)

    logger.info("Loading evaluation dataset.")
    dataset = load_data(args.i)

    outside = dataset.tag_dict.get_label_id("O")

    logger.info("Evaluating: simple accuracy per type\n")
    correct_type, incorrect_type = results_type(dataset, outside)
    acc_type = accuracy(correct_type, incorrect_type)
    acc = score_overall(acc_type)
    print(acc)
    print("\n")

    logger.info("Evaluating: precision, recall and fscore\n")
    tp, fp, tn, fn = results(dataset, outside)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    print("Precision: {}\nRecall: {}\nFscore: {}\n".format(prec, rec, fscore(prec, rec)))

    logger.info("Evaluating: precision, recall and fscore per type, skipping 0 Fscore\n")
    counts_per_type = results_type_fscore(dataset, outside)
    fscore_per_type = fscore_type(counts_per_type)
    f = score_overall(fscore_per_type, skip_zero=True)
    print("Fscore: {}\n".format(f))


