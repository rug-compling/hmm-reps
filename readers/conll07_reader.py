from readers.dependency_instance import DependencyInstance


class Conll07Reader:
    # ## read Conll 2007 data: reusing https://github.com/bplank/myconllutils
    ### http://ilk.uvt.nl/conll/index.html#dataformat

    def __init__(self, filename):
        self.filename = filename
        self.startReading()

    def __iter__(self):
        i = self.getNext()
        while i:
            yield i
            i = self.getNext()

    def startReading(self):
        self.FILE = open(self.filename, "r")

    def getNext(self):
        # return next instance or None

        line = self.FILE.readline()

        line = line.strip()
        lineList = line.split("\t")

        ids = []
        form = []
        lemma = []
        cpos = []
        pos = []
        feats = []
        head = []
        deprel = []
        phead = []
        pdeprel = []

        if len(lineList) >= 12:  #CONLL 2009 format
            while len(lineList) >= 12:
                ids.append(int(lineList[0]))
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[5])
                pos.append(lineList[4])
                feats.append(lineList[6])
                head.append(int(lineList[8]))
                deprel.append(lineList[10])
                phead.append(lineList[9])
                pdeprel.append(lineList[11])

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        elif len(lineList) == 10:
            # contains all cols, also phead/pdeprel
            while len(lineList) == 10:
                ids.append(int(lineList[0]))
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[3])
                pos.append(lineList[4])
                feats.append(lineList[5])
                head.append(int(lineList[6]))
                deprel.append(lineList[7])
                phead.append(lineList[8])
                pdeprel.append(lineList[9])

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        elif len(lineList) == 8:
            while len(lineList) == 8:
                ids.append(lineList[0])
                form.append(lineList[1])
                lemma.append(lineList[2])
                cpos.append(lineList[3])
                pos.append(lineList[4])
                feats.append(lineList[5])
                head.append(int(lineList[6]))
                deprel.append(lineList[7])
                phead.append("_")
                pdeprel.append("_")

                line = self.FILE.readline()
                line = line.strip()
                lineList = line.split("\t")
        elif len(lineList) > 1:
            raise Exception("not in right format!")

        if len(form) > 0:
            return DependencyInstance(ids, form, lemma, cpos, pos, feats, head, deprel, phead, pdeprel)
        else:
            return None


    def getInstances(self):
        instance = self.getNext()

        instances = []
        while instance:
            instances.append(instance)

            instance = self.getNext()
        return instances

    def getSentences(self):
        """ return sentences as list of lists """
        instances = self.getInstances()
        sents = []
        for i in instances:
            sents.append(i.form)
        return sents

    def getStrings(self, wordform="form"):
        """ sentence is one space-separated string in a list """
        if wordform == "lemma":
            return (" ".join(instance.lemma) for instance in self)
        else:
            return (" ".join(instance.form) for instance in self)

    def writeStrings(self, filepath, wordform="form"):
        """ write form to output. """
        with open(filepath, "w") as out:
            for i in self.getStrings(wordform=wordform):
                out.write("{}\n".format(i))

    def getVocabulary(self, n_sent=float("Inf"), add_root=True, lemmas=False):
        """
         vocabulary with frequencies
         :param n_sent: max number of sentences to consider
         :param add_root: add artificial symbol *root* to vocab
         :param lemmas: use lemma instead of form
        """
        from collections import defaultdict

        vocab = defaultdict(int)
        instance = self.getNext()
        c = 1
        if lemmas:
            while instance and (c <= n_sent):
                for w in instance.getSentenceLemmas():
                    vocab[w] += 1
                vocab["*root*"] += 1
                instance = self.getNext()
                c += 1
        else:
            while instance and (c <= n_sent):
                for w in instance.getSentence():
                    vocab[w] += 1
                vocab["*root*"] += 1
                instance = self.getNext()
                c += 1
        return vocab

    def getRelationVocabulary(self, n_sent=float("Inf")):
        """
         vocabulary of relation labels
         :param n_sent: max number of sentences to consider
         :param add_root: add artificial symbol *root* to vocab
        """
        vocab = set()
        instance = self.getNext()
        c = 1
        while instance and (c <= n_sent):
            vocab.update(instance.deprel)
            instance = self.getNext()
            c += 1
        return vocab

    def getCorpusTriples(self, wordform="form"):
        """ gets counts of head_w\tdep_w occurences """
        from collections import defaultdict

        counts = defaultdict(int)

        if wordform == "form":
            for instance in self:
                for i in instance.getBareFormTriples():
                    counts[i] += 1
        elif wordform == "lemma":
            for instance in self:
                for i in instance.getBareLemmaTriples():
                    counts[i] += 1
        return counts

    def getkCorpusTriples(self, k):
        """ gets counts of head_w\tdep_w occurences for k instances """
        from collections import defaultdict

        counts = defaultdict(int)

        for instance in self:
            if k > 0:
                for i in instance.getBareFormTriples():
                    counts[i] += 1
                k -= 1
            else:
                break
        return counts


def writeCorpusTriples(counts, filepath):
    with open(filepath, "w") as out:
        for k, v in counts.items():
            out.write("{0[0]}\t{0[1]}\t{1}\n".format(k.split("\t"), v))


def filter_freq(s, f, vocab, lemma=False):
    if s is not None:
        for i in range(len(s)):
            if lemma:
                if vocab[s.lemma[i]] < f:
                    s.lemma[i] = "*unk*"
            else:
                if vocab[s.form[i]] < f:
                    s.form[i] = "*unk*"
        return s
    return


def filter_len(s, min_len, max_len):
    if s is not None:
        if min_len < len(s) < max_len:
            return s
    return
