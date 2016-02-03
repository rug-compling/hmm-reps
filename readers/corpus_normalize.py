from readers.conll07_reader import Conll07Reader, filter_len, filter_freq
from readers.vocab import read_vocab_freq, read_vocab


class CorpusNormalizeChain:
    def __init__(self):

        self.vocab_file = "vocab"
        self.vocab = read_vocab(self.vocab_file)

        self.corpus_file = "heldout"
        #outfile
        self.corpus_file_normalized = "heldout.norm"

    def unk(self, freq=1):
        """
        Replace low-frequent words with *unk* in order to perform smoothing.

        :param freq: words with that freq will be replaced
        """
        with open(self.corpus_file) as infile, open("{}.unk{}".format(self.corpus_file_normalized, freq),
                                                    "w") as outfile:
            for l in infile:
                newline = []
                for w in l.strip().split(" "):
                    if self.vocab[w] <= freq:
                        newline.append("*unk*")
                    else:
                        newline.append(w)
                if newline:
                    outfile.write(" ".join(newline) + "\n")


class CorpusNormalizeTree:
    """conll format"""

    def __init__(self, lemmas=True, length_only=False):
        self.length_only = length_only
        self.vocab = None
        if not self.length_only:
            self.vocab_file = "vocab"  # file of form: w\tf\n
            self.vocab = read_vocab_freq(self.vocab_file)
        self.corpus_file = "SONAR.tokenlevel"
        #outfile
        self.corpus_file_normalized = "{}.norm".format(self.corpus_file)
        self.lemmas = lemmas

    def unk_conll(self, freq=1):
        """
        quicky and dirty way of introducing *unk* to conll
        """
        w_idx = 2 if self.lemmas else 1  # lemma or word column index in conll
        with open(self.corpus_file) as infile, open("{}.unk{}".format(self.corpus_file_normalized, freq),
                                                    "w") as outfile:
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

    def filter_sents(self, freq=1, min_len=4, max_len=40):
        filepath = "{0}.unk{1}".format(self.corpus_file_normalized,
                                       freq) if not self.length_only else "{0}.length".format(
            self.corpus_file_normalized)
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


if __name__ == "__main__":
    c = CorpusNormalizeTree(lemmas=False)
    c.filter_sents(freq=40)
