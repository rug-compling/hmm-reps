import argparse

from readers.conll07_reader import Conll07Reader, filter_len, filter_freq
from readers.conll_corpus import ConllCorpus
from readers.text_corpus import TextCorpus
from readers.vocab import read_vocab_freq, read_vocab


class CorpusNormalizeChain:
    def __init__(self, dataset, output):
        self.vocab = self.create_vocab(dataset)
        self.corpus_file = dataset
        #outfile
        self.corpus_file_unk = output

    def unk(self, freq=1):
        """
        Replace low-frequent words with *unk* in order to perform smoothing.

        :param freq: words with that freq will be replaced
        """
        with open(self.corpus_file) as infile, open(self.corpus_file_unk, "w") as outfile:
            for l in infile:
                newline = []
                for w in l.strip().split(" "):
                    if self.vocab[w] <= freq:
                        newline.append("*unk*")
                    else:
                        newline.append(w)
                if newline:
                    outfile.write(" ".join(newline) + "\n")

    def create_vocab(self, dataset):
        d = TextCorpus(dataset, howbig=1e10)
        return d.x_dict


class CorpusNormalizeTree:
    """conll format"""

    def __init__(self, dataset, output, lemmas=True, length_only=False):
        self.length_only = length_only
        self.vocab = self.create_vocab(dataset, lemmas=lemmas)
        self.corpus_file = dataset
        self.corpus_file_unk = output
        self.lemmas = lemmas

    def unk(self, freq):
        """
        quicky and dirty way of introducing *unk* to conll
        """
        w_idx = 2 if self.lemmas else 1  # lemma or word column index in conll
        with open(self.corpus_file) as infile, open(self.corpus_file_unk, "w") as outfile:
            for c, l in enumerate(infile, 1):
                if l.strip() == "":
                    outfile.write(l)
                else:
                    splitted = l.split("\t")
                    if self.vocab[splitted[w_idx]] <= freq:
                        splitted[w_idx] = "*unk*"
                        outfile.write("\t".join(splitted))
                    else:
                        outfile.write(l)

    def filter_sents(self, freq, min_len, max_len):
        filepath = self.corpus_file_unk+"filt"
        with open(filepath, "w") as out:
            reader = Conll07Reader(self.corpus_file)
            s = reader.getNext()
            while s:
                if self.length_only:
                    s = filter_len(s, min_len, max_len)
                else:
                    s = filter_freq(filter_len(s, min_len, max_len), freq, self.vocab, lemma=self.lemmas)
                if s is not None:
                    s.writeout_handle(out)
                s = reader.getNext()

    def num_conll(self):
        """
        replace number occurrences with *num*
        """
        pass

    def create_vocab(self, dataset, lemmas):
        d = ConllCorpus(dataset, howbig=1e10, lemmas=lemmas)
        return d.x_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conll", action='store_true', default=False, help="conll format")
    parser.add_argument("-d", "--dataset", required=True, help="corpus file")
    parser.add_argument("-o", "--output", required=True, help="unk'ed corpus file")
    parser.add_argument("--freq_thresh", type=int, default=1, required=True, help="replace all words occuring #freq_thresh of times with *unk*")
    parser.add_argument("--lemmas", action='store_true', default=False, help="use lemma instead of word column in conll")
    parser.add_argument("--min_len", type=int, default=4, help="minimum sentence length for filter_sent method")
    parser.add_argument("--max_len", type=int, default=40, help="maximum sentence length for filter_sent method")
    args = parser.parse_args()

    if args.conll:
        c = CorpusNormalizeTree(args.dataset, args.output, lemmas=args.lemmas)
        #c.filter_sents(freq=args.freq_thresh, min_len=args.min_len, max_len=args.max_len)
    else:
        c = CorpusNormalizeChain(args.dataset, args.output)
    c.unk(freq=args.freq_thresh)

    print("Finished normalizing.")