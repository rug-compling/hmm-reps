import sys
import os

from readers.conll_corpus import ConllCorpus
from readers.conll_files_index import ConllFilesIndex
from sequences.label_dictionary import LabelDictionary
from sequences.sequence_list_label import SequenceListLabel
from trees.tree import Tree


data_dir = "eval/ner/data/"

ned_train = "{}/ned.train.lemma".format(data_dir)
ned_train_parsed = "{}/ned.train.parsed".format(data_dir)
ned_train_parsed_files_path = "{}/ned.train.xml/".format(data_dir)
ned_dev = "{}/ned.testa.lemma".format(data_dir)
ned_dev_parsed = "{}/ned.testa.parsed".format(data_dir)
ned_dev_parsed_files_path = "{}/ned.testa.xml/".format(data_dir)
ned_test = "{}/ned.testb.lemma".format(data_dir)
ned_test_parsed = "{}/ned.testb.parsed".format(data_dir)
ned_test_parsed_files_path = "{}/ned.testb.xml/".format(data_dir)


def get_words_from_tree(tree, trees_vocab):
    return [trees_vocab.get_label_name(node.name) for node in tree if not node.is_root()]


class Conll2002NerCorpus():
    """
    Optionally reads text to which we want to apply a wordrep such as hmm.
    - no update of the wordrep_dict; every word not in it (from x_dict),
    gets *unk* id needed for successful decoding
    "
    """

    def __init__(self, wordrep_dict=None, eval_spec_rel=False, dirname=None, lr=False, use_wordrep_tree=False):
        """
        :param wordrep_dict: x_dictionary from training of word representations
        :param use_wordrep_tree: use parse tree representations
        """

        self.wordrep_dict = wordrep_dict
        if self.wordrep_dict is not None:
            self.word_dict = self.wordrep_dict.copy()
        else:
            self.word_dict = LabelDictionary()
        self.tag_dict = LabelDictionary()  # ner tag
        self.use_wordrep_tree = use_wordrep_tree
        self.sequence_list = None  # SequenceListLabel(self.word_dict, self.tag_dict, self.wordrep_dict)
        self.eval_spec_rel = eval_spec_rel
        self.dirname = dirname
        self.lr = lr
        # for conll2002 lemma format preparation:
        self.tree_vocab = None

    def read_sequence_list_conll(self, train_file, train_file_parsed=None, train_files_parsed_path=None,
                                 max_sent_len=100000, max_nr_sent=100000):
        """
        Read a conll2002 or conll2003 file into a sequence list.
        Optionally add a sequence list/tree with *unk* for decoding in wordrep.
        """
        instance_list = self.read_conll_instances(train_file, train_file_parsed, train_files_parsed_path, max_sent_len,
                                                  max_nr_sent)

        if self.wordrep_dict is not None:

            seq_list = SequenceListLabel(self.word_dict, self.tag_dict, self.wordrep_dict)  # for indices
            for sent_x, sent_y, sent_ in instance_list:
                # sent_ is a normalized tree
                if self.use_wordrep_tree:
                    seq_list.add_sequence(sent_x, sent_y, None, sent_)
                # sent is a normalized chain
                else:
                    seq_list.add_sequence(sent_x, sent_y, sent_)
        else:
            seq_list = SequenceListLabel(self.word_dict, self.tag_dict)  # for indices
            for sent_x, sent_y in instance_list:
                seq_list.add_sequence(sent_x, sent_y)

        return seq_list

    def read_conll_instances(self, file, file_parsed, files_parsed_path, max_sent_len, max_nr_sent):
        """
        TODO: refactor the entire method, lots of overlap chain/tree/token/lemma
        """

        def get_tree(n_inst):
            trees = ConllCorpus(file_parsed, howbig=1000000, lemmas=True, eval_spec_rels=self.eval_spec_rel,
                                dirname=self.dirname, lr=self.lr)
            trees.prepare_trees()
            # not every instance has a corresponding tree due to errors in parsing
            conll_idx = ConllFilesIndex(files_parsed_path)
            conll_idx.create_ids_set()
            # extend instances with trees
            c_append = 0
            for i in range(n_inst):
                # we have a parse:
                if i + 1 in conll_idx.fileids:
                    inst = self.normalize_tree(trees.train[c_append], trees.x_dict, c_append)
                    c_append += 1
                # we don't have a parse:
                else:
                    inst = None
                yield inst

        if self.use_wordrep_tree:
            if file_parsed is None or files_parsed_path is None:
                sys.exit("Missing parsed file.")

        contents = open(file, encoding="iso-8859-1")
        nr_sent = 0
        instances = []
        ex_x = []
        ex_y = []
        include_ex_z = (self.wordrep_dict is not None and not self.use_wordrep_tree)
        if include_ex_z:
            ex_z = []

        for line in contents:
            if line.startswith("-DOCSTART"):
                continue
            toks = line.split()
            if len(toks) < 3:
                if 0 < len(ex_x) < max_sent_len:  # len(ex_x) > 1 # escape one-word sentences
                    nr_sent += 1
                    instances.append([ex_x, ex_y, ex_z] if include_ex_z else [ex_x, ex_y])
                if nr_sent >= max_nr_sent:
                    break
                ex_x = []
                ex_y = []
            else:
                tag = toks[2]
                word = toks[0]
                if word not in self.word_dict:
                    self.word_dict.add(word)
                if tag not in self.tag_dict:
                    self.tag_dict.add(tag)
                ex_x.append(word)
                ex_y.append(tag)
                if include_ex_z:
                    ex_z.append(self.normalize_word(word))

        # add parsed data to use tree wordreps
        if self.use_wordrep_tree:
            for c, instance in enumerate(get_tree(len(instances))):
                # get parsed data
                inst = instance
                instances[c].append(inst)

        return instances  # try generator

    def prepare_lemmatized_conll2002(self, train_file, train_file_parsed=None, train_files_parsed_path=None,
                                     output_f=None):
        self.use_wordrep_tree = True  # need parsed data
        docstarts, instances = self.prepare_conll_instances(train_file, train_file_parsed, train_files_parsed_path)
        if output_f is None:
            return instances
        else:
            header = "-DOCSTART- -DOCSTART- O"
            with open(output_f, "w") as outfile:
                for n, instance in enumerate(instances):
                    # doc headers
                    if n in docstarts:
                        outfile.write("{}\n".format(header))
                    if isinstance(instance, list):
                        for _, postag, tag, lemma in zip(*instance):
                            outfile.write("{} {} {}\n".format(lemma, postag, tag))
                        outfile.write("\n")
                    else:
                        sys.exit("invalid instance")

    def prepare_conll_instances(self, file, file_parsed, files_parsed_path):
        def get_tree(n_inst):
            trees = ConllCorpus(file_parsed, howbig=1000000, lemmas=True, eval_spec_rels=self.eval_spec_rel,
                                dirname=self.dirname, lr=self.lr)
            trees.prepare_trees()
            self.tree_vocab = trees.x_dict
            # not every instance has a corresponding tree due to errors in parsing
            conll_idx = ConllFilesIndex(files_parsed_path)
            conll_idx.create_ids_set()
            # extend instances with trees
            c_append = 0
            for i in range(n_inst):
                # we have a parse:
                if i + 1 in conll_idx.fileids:
                    inst = trees.train[c_append]
                    c_append += 1
                # we don't have a parse:
                else:
                    inst = None
                yield inst

        max_sent_len = 1000000
        max_nr_sent = 1000000
        if file_parsed is None or files_parsed_path is None:
            sys.exit("Missing parsed file.")

        contents = open(file, encoding="iso-8859-1")
        nr_sent = 0
        instances = []
        ex_x = []
        ex_x_pos = []
        ex_y = []
        docstarts = set()  # track docstarts header

        for line in contents:
            if line.startswith("-DOCSTART"):
                docstarts.add(nr_sent)
                continue
            toks = line.split()
            if len(toks) < 3:
                if 0 < len(ex_x) < max_sent_len:  # len(ex_x) > 1 # escape one-word sentences
                    nr_sent += 1
                    instance = [ex_x, ex_x_pos, ex_y]
                    instances.append(instance)
                if nr_sent >= max_nr_sent:
                    break
                ex_x = []
                ex_x_pos = []
                ex_y = []
            else:
                tag = toks[2]
                postag = toks[1]
                word = toks[0]
                ex_x.append(word)
                ex_x_pos.append(postag)
                ex_y.append(tag)

        for c, instance in enumerate(get_tree(len(instances))):
            ex_z = self.get_words(instance, self.tree_vocab)  # should get lemmas (from ConllCorpus)
            if ex_z is None:
                inst = [i for i in instances[c][0]]
                print("None instance")
            else:
                assert len(ex_z) == len(instances[c][0])
                inst = ex_z
            instances[c].append(inst)

        return docstarts, instances  # try generator

    def normalize_word(self, word):
        if word not in self.wordrep_dict:
            return "*unk*" if word.lower() not in self.wordrep_dict else word.lower()
        else:
            return word

    def normalize_tree(self, tree, trees_vocab, c):
        """
        Recode the name index based on wordrep_dict.
        Modify tree.name such that *unk* or lowercase words are included.
        """
        for node in tree:
            w = trees_vocab.get_label_name(node.name)
            # if c==0:
            #    print("{}\t{}".format(w, self.normalize_word(w)))
            new_name = self.wordrep_dict.get_label_id(self.normalize_word(w))
            node.set_name(new_name)
        return tree

    def get_words(self, instance, vocab):
        if isinstance(instance, Tree):
            return get_words_from_tree(instance, vocab)
        print("None instance in Conll2002NerCorpus")
        return None

    def write_conll_instances(self, gold, predictions, file, sep=" "):
        """
        Create dataset with appended predictions as the last column.
        """
        assert len(gold) == len(predictions)
        contents = open(file, "w", encoding="iso-8859-1")
        for gold_seq, pred_seq in zip(gold.seq_list, predictions):
            for x, y, y_hat in zip(gold_seq.x, gold_seq.y, pred_seq.y):
                contents.write("{}{sep}{}{sep}{}\n".format(gold_seq.sequence_list.x_dict.get_label_name(x),
                                                           gold_seq.sequence_list.y_dict.get_label_name(y),
                                                           pred_seq.sequence_list.y_dict.get_label_name(y_hat),
                                                           sep=sep))
            contents.write("\n")

    # # Dumps a corpus into a file
    def save_corpus(self, dirname):
        if not os.path.isdir(dirname + "/"):
            os.mkdir(dirname + "/")
        #word_fn = open(dir+"word.dic","w")
        #for word_id,word in enumerate(self.int_to_word):
        #    word_fn.write("{}\t{}\n".format(word_id, word))
        #word_fn.close()
        #tag_fn = open(dir+"tag.dic","w")
        #for tag_id,tag in enumerate(self.int_to_tag):
        #    tag_fn.write("{}\t{}\n".format(tag_id, tag))
        #tag_fn.close()
        #word_count_fn = open(dir+"word.count","w")
        #for word_id,counts in self.word_counts.iteritems():
        #    word_count_fn.write("{}\t{}\n".format(word_id,counts))
        #word_count_fn.close()
        self.sequence_list.save(dirname + "sequence_list")

    ## Loads a corpus from a file
    def load_corpus(self, dirname):
        word_fn = open(dirname + "word.dic")
        for line in word_fn:
            word_nr, word = line.strip().split("\t")
            self.int_to_word.append(word)
            self.word_dict[word] = int(word_nr)
        word_fn.close()
        tag_fn = open(dirname + "tag.dic")
        for line in tag_fn:
            tag_nr, tag = line.strip().split("\t")
            if tag not in self.tag_dict:
                self.int_to_tag.append(tag)
                self.tag_dict[tag] = int(tag_nr)
        tag_fn.close()
        word_count_fn = open(dirname + "word.count")
        for line in word_count_fn:
            word_nr, word_count = line.strip().split("\t")
            self.word_counts[int(word_nr)] = int(word_count)
        word_count_fn.close()
        self.sequence_list.load(dirname + "sequence_list")


if __name__ == "__main__":
    c = Conll2002NerCorpus(use_wordrep_tree=True)
    c.prepare_lemmatized_conll2002(ned_test, ned_test_parsed, ned_test_parsed_files_path, "{}.lemma".format(ned_test))
    c = Conll2002NerCorpus(use_wordrep_tree=True)
    c.prepare_lemmatized_conll2002(ned_dev, ned_dev_parsed, ned_dev_parsed_files_path, "{}.lemma".format(ned_dev))
    c = Conll2002NerCorpus(use_wordrep_tree=True)
    c.prepare_lemmatized_conll2002(ned_train, ned_train_parsed, ned_train_parsed_files_path,
                                   "{}.lemma".format(ned_train))