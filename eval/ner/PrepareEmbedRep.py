import sys

from eval.ner.readers.Conll2002NerCorpus import Conll2002NerCorpus, ned_train, ned_dev, ned_test
from eval.ner.readers.Conll2003NerCorpus import Conll2003NerCorpus, eng_train, eng_dev, eng_test, muc_test
from eval.ner.readers.word2vec import load_embed
from readers.vocab import read_vocab
from sequences.label_dictionary import LabelDictionary


class PrepareEmbedRep():
    """
    Applying word embeddings to the evaluation dataset.
    """

    def __init__(self, embed, embed_v, lang, use_wordrep_tree=False, logger=None, use_muc=False):

        self.embeddings = load_embed(embed, embed_v)
        self.lang = lang
        self.use_wordrep_tree = use_wordrep_tree
        self.logger = logger
        self.ner_corpus = None
        self.use_muc = use_muc

        if not self.use_wordrep_tree:
            if self.lang == "nl":
                self.train_seq, self.dev_seq, self.test_seq = self.prepare_seqs_nl(embed_v)
            elif self.lang == "en":
                self.train_seq, self.dev_seq, self.test_seq, self.muc_seq = self.prepare_seqs_en(embed_v)
            else:
                sys.exit("Unrecognized language argument.")
        else:
            sys.exit("Works only on chains, not trees.")

    def prepare_seqs_nl(self, vocab_f):
        self.ner_corpus = Conll2002NerCorpus(wordrep_dict=LabelDictionary(read_vocab(vocab_f)))

        train_seq = self.ner_corpus.read_sequence_list_conll(ned_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(ned_dev)
        test_seq = self.ner_corpus.read_sequence_list_conll(ned_test)

        mapper_corpus(train_seq, self.embeddings)
        mapper_corpus(dev_seq, self.embeddings)
        mapper_corpus(test_seq, self.embeddings)

        return train_seq, dev_seq, test_seq

    def prepare_seqs_en(self, vocab_f):
        self.ner_corpus = Conll2003NerCorpus(wordrep_dict=LabelDictionary(read_vocab(vocab_f)))

        train_seq = self.ner_corpus.read_sequence_list_conll(eng_train)
        dev_seq = self.ner_corpus.read_sequence_list_conll(eng_dev)
        test_seq = self.ner_corpus.read_sequence_list_conll(eng_test)
        muc_seq = self.ner_corpus.read_sequence_list_conll(muc_test) if self.use_muc else None

        mapper_corpus(train_seq, self.embeddings)
        mapper_corpus(dev_seq, self.embeddings)
        mapper_corpus(test_seq, self.embeddings)
        if self.use_muc:
            mapper_corpus(muc_seq, self.embeddings)

        return train_seq, dev_seq, test_seq, muc_seq


def mapper_corpus(dataset, emb):
    """Run posterior_decode at corpus level."""
    for seq in dataset.seq_list:
        seq.w = emb[seq.z]


if __name__ == "__main__":
    p = PrepareEmbedRep(
        embed="embeddings",
        embed_v="embeddings.vocab", lang="en")