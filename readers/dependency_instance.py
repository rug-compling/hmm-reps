import sys

__author__ = 'sim'


class DependencyInstance:
    def __init__(self, ids, form, lemma, cpos, pos, feats, headid, deprel, phead, pdeprel):
        self.ids = ids
        self.form = form
        self.lemma = lemma
        self.cpos = cpos
        self.pos = pos
        self.feats = feats
        self.headid = headid
        self.deprel = deprel
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        s = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n"
        sout = ""
        for i in range(len(self.form)):
            sout += s.format(self.ids[i], self.form[i], self.lemma[i], self.cpos[i], self.pos[i], self.feats[i],
                             self.headid[i], self.deprel[i], self.phead[i], self.pdeprel[i])
        return sout

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield (self.ids[i], self.form[i], self.lemma[i], self.cpos[i], self.pos[i], self.feats[i], self.headid[i],
                   self.deprel[i], self.phead[i], self.pdeprel[i])

    def writeout(self, filepath):
        if self is not None:
            with open(filepath, "a") as out:
                out.write(self.__str__())
                out.write("\n")

    def writeout_handle(self, fileh):
        if self is not None:
            fileh.write(self.__str__())
            fileh.write("\n")

    def equalForm(self, instance):
        for f1, f2 in zip(self.form, instance.form):
            if f1 != f2:
                return False
        return True

    def equalHeads(self, instance):
        for f1, f2 in zip(self.headid, instance.headid):
            if f1 != f2:
                return False
        return True

    def containsRelation(self, label):
        return label in self.deprel

    def containsWord(self, word):
        return word in self.form

    def containsPostag(self, postag):
        return postag in self.pos

    def equalLabels(self, instance):
        for f1, f2 in zip(self.deprel, instance.deprel):
            if f1 != f2:
                return False
        return True

    def equalHeadsAndLabels(self, instance):
        return self.equalHeads(instance) and self.equalLabels(instance)

    def getSentenceLength(self):
        return len(self.form)

    def getSentence(self):
        return self.form

    def getIds(self):
        return self.ids

    def getHeads(self):
        """
         head of id
        """
        return zip(self.ids, self.headid)

    def getSentenceLemmas(self):
        return self.lemma

    def getSentenceByPos(self):
        sentence = []
        for i in range(len(self.pos)):
            if self.pos[i] in ["NOM", "NAM", "ABR", "VER", "PRP", "NUM", "ADV", "ADJ", "VER:subp", "VER:futu",
                               "VER:simp", "VER:subi", "VER:pres", "VER:cond", "VER:ppre", "VER:impf"]:
                sentence.append(self.lemma[i])
        return sentence

    def getLemmaTriples(self):
        return self.getTriples(self.lemma)

    def getFormTriples(self):
        return self.getTriples(self.form)

    def getBareFormTriples(self):
        return self.getBareTriples(self.form)

    def getBareLemmaTriples(self):
        return self.getBareTriples(self.lemma)

    def getBareTriples(self, wordform):
        """ no counts. no rel. tab-sep. no root triple"""
        for i in range(len(wordform)):
            w_d = wordform[i]
            hid = self.headid[i]
            if hid != 0:
                w_h = wordform[hid - 1]
                if type(w_h) != str:
                    print(wordform)
            else:
                continue
            yield "{}\t{}".format(w_h, w_d)

    def getTriples(self, wordform):
        triples = {}
        for i in range(len(wordform)):
            r = self.deprel[i]
            w_d = wordform[i].replace(" ", "")
            hid = self.headid[i]
            if hid != 0:
                w_h = wordform[hid - 1].replace(" ", "")
            else:
                w_h = '<root-LEMMA>'
            triple = "{} {} {}".format(r, w_h, w_d)
            triples[triple] = triples.get(triple, 0) + 1
        return triples

    def getAllLemmaTriples(self):
        return self.getAllTriples(self.lemma)

    def getAllFormTriples(self):
        return self.getAllTriples(self.form)

    def getAllTriples(self, wordform):
        """ also returns counts of parts of relation """
        triples = self.getTriples(wordform)
        actualtriples = triples.copy()
        for triple in actualtriples:
            try:
                r, w_h, w_d = triple.split(" ")

                triple_r_w1 = "{} {} ".format(r, w_h)
                triples[triple_r_w1] = triples.get(triple_r_w1, 0) + 1

                triple_w2x = "{} {}".format(r, w_d)
                triples[triple_w2x] = triples.get(triple_w2x, 0) + 1

            except ValueError:
                print("Error when splitting triples: {}".format(triple))
                sys.exit(-1)
        return triples