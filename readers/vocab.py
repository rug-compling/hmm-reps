from util.util import line_reader


def read_vocab(vocab_f, minfreq=0):
    vocab = []
    for l in line_reader(vocab_f):
        if eval(l.rstrip().split("\t")[1]) > minfreq:
            vocab.append(l.rstrip().split("\t")[0])

    return vocab


def read_vocab_freq(vocab_f):
    vocab = {}
    for l in line_reader(vocab_f):
        vocab[l.rstrip().split("\t")[0]] = eval(l.rstrip().split("\t")[1])

    return vocab
