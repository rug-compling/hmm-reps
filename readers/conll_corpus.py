import sys

from readers.conll07_reader import Conll07Reader
from readers.vocab import read_vocab
from sequences.label_dictionary import LabelDictionary
from sequences.relation_dictionary import RelationDictionary
from trees.bpedge import BPEdge
from trees.bpnode import BPNode
from trees.bptree import BPTree
from trees.tree_list import TreeList

__author__ = 'sim'


class ConllCorpus:
    def __init__(self, corpus_file, minfreq=0, howbig=1000, lemmas=True, spec_rels=None, dirname=None,
                 eval_spec_rels=False, lr=False):

        """
        :param howbig: number of sentences to take into account
        """
        self.corpus_file = corpus_file
        self.vocab_file = "{}.vocab{}".format(self.corpus_file, howbig)
        self.rel_file = "{}.rels.vocab{}".format(self.corpus_file, howbig)  # dependency labels

        self.minfreq = minfreq
        self.howbig = howbig
        self.lemmas = lemmas
        self.lr = lr
        #read built vocab
        try:
            self.x_dict = LabelDictionary(read_vocab(self.vocab_file, self.minfreq))
        #except FileNotFoundError:
        except IOError:
            self.prepare_vocab_dict()
            self.x_dict = LabelDictionary(read_vocab(self.vocab_file, self.minfreq))
        print("LabelDictionary created.")

        if eval_spec_rels:  # in evaluation
            try:
                import pickle

                self.r_dict = pickle.load(open("{}/r_dict.pickle".format(dirname), "rb"))
            except IOError:
                sys.exit("r_dict does not exist.")
        else:
            if self.lr:
                self.r_dict = RelationDictionary(["left", "right"])
                self.r_dict.write("{}/r_dict.pickle".format(dirname))
            else:
                try:
                    r_dict = LabelDictionary([l.strip() for l in open(self.rel_file)])
                except IOError:
                    self.prepare_rel_vocab_dict()
                    r_dict = LabelDictionary([l.strip() for l in open(self.rel_file)])
                if spec_rels:
                    self.r_dict = RelationDictionary(spec_rels)
                    self.r_dict.add("OTHER")
                    self.r_dict.add_fixed_id((set(r_dict.names) - set(spec_rels)), self.r_dict.get_label_id("OTHER"))
                    self.r_dict.write("{}/r_dict.pickle".format(dirname))
                else:
                    self.r_dict = r_dict
        print("Relation/LabelDictionary created.")

    def prepare_trees(self):
        self.train = TreeList()
        #print(self.train)
        reader = Conll07Reader(self.corpus_file)
        sent = reader.getNext()
        c = 1
        while sent and (c <= self.howbig):
            t = self.prepare(sent, lr=self.lr)
            if t is not None:
                self.train.add_tree(t)
                #tracker.create_snapshot()
            #tracker.stats.print_summary()
            sent = reader.getNext()
            c += 1

    def prepare_trees_gen(self):
        reader = Conll07Reader(self.corpus_file)
        sent = reader.getNext()
        c = 1
        while sent and (c <= self.howbig):
            t = self.prepare(sent, lr=self.lr)
            if t is not None:
                yield t
                #tracker.create_snapshot()
            #tracker.stats.print_summary()
            sent = reader.getNext()
            c += 1

    def prepare(self, sent, lr=False):
        t = BPTree()
        #tracker = ClassTracker()
        #tracker.track_object(t)
        #tracker.create_snapshot()
        #1.pass: create nodes
        elems = sent.getSentenceLemmas() if self.lemmas else sent.getSentence()

        if lr:
            for w, i in zip(elems, sent.getIds()):
                idx = self.x_dict.get_label_id(w)
                t.add_node(BPNode(i, idx))
        else:
            for w, i, r in zip(elems, sent.getIds(), sent.deprel):
                idx = self.x_dict.get_label_id(w)
                ridx = self.r_dict.get_label_id(r)
                t.add_node(BPNode(i, idx, rel=ridx))
        #add root
        #tracker.create_snapshot("add words of sent")
        idx = self.x_dict.get_label_id("*root*")
        t.add_node(BPNode(0, idx))
        #tracker.create_snapshot("add ROOT")
        #2.pass: create edges
        seen = set()  # catch direct loops
        for i, i_head in sent.getHeads():
            # this only catches direct loops; TODO: use is_acyclic check
            if (i, i_head) in seen or (i_head, i) in seen:
                print("Tree with loop caught")
                t = None
                break
            else:
                seen.add((i, i_head))
            if i == i_head:  # not allowed
                print("Skipping sentence: parent is its own child")
                t = None
                break
            parent = t[i_head]
            child = t[i]
            if lr:
                child.rel = self.r_dict.get_label_id("left") if i_head > i else self.r_dict.get_label_id(
                    "right")  #w occurs left/right of its parent
            if parent is None or child is None:
                print()
            edge = BPEdge(parent, child)
            t.add_edge(edge)
            #tracker.create_snapshot("add edge")
            t.add_edge_to_map(parent, child, edge)
            #tracker.create_snapshot("add edge to map")

        return t

    def prepare_vocab_dict(self):
        reader = Conll07Reader(self.corpus_file)
        vocab_dict = reader.getVocabulary(n_sent=self.howbig, add_root=True, lemmas=self.lemmas)

        with open(self.vocab_file, "w") as OUT:
            for w, f in vocab_dict.items():
                OUT.write("{}\t{}\n".format(w, f))

        print("Vocabulary file prepared.")

    def prepare_rel_vocab_dict(self):
        reader = Conll07Reader(self.corpus_file)
        vocab = reader.getRelationVocabulary(n_sent=self.howbig)

        with open(self.rel_file, "w") as OUT:
            for r in vocab:
                OUT.write("{}\n".format(r))

        print("Relation vocabulary file prepared.")

#if __name__ == "__main__":
#    ConllCorpus("SONAR_random4000000.roots.1M.train.conll", howbig=100000)
