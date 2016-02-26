import sys

from eval.ner.readers.Conll2002NerCorpus import Conll2002NerCorpus
from readers.conll_corpus import ConllCorpus


data_dir = "eval/ner/data/"

eng_train = "{}eng.train".format(data_dir)
eng_train_parsed = "{}eng.train.parsed".format(data_dir)
eng_dev = "{}eng.testa".format(data_dir)
eng_dev_parsed = "{}eng.testa.parsed".format(data_dir)
eng_test = "{}eng.testb".format(data_dir)
eng_test_parsed = "{}eng.testb.parsed".format(data_dir)
muc_test = "{}MUC7.NE.formalrun.sentences.columns.gold".format(data_dir)
muc_test_parsed = "{}MUC7.NE.formalrun.sentences.columns.gold.parsed".format(data_dir)


class Conll2003NerCorpus(Conll2002NerCorpus):
    def read_conll_instances(self, file, file_parsed, files_parsed_path, max_sent_len, max_nr_sent):
        def get_tree(n_inst):
            trees = ConllCorpus(file_parsed, howbig=1000000, lemmas=False, eval_spec_rels=self.eval_spec_rel,
                                dirname=self.dirname, lr=self.lr)
            trees.prepare_trees()
            # extend instances with trees
            assert len(trees.train) == n_inst, "Number of parses not equal to number of classification instances."
            c_append = 0
            for i in range(n_inst):
                # we have a parse:
                inst = self.normalize_tree(trees.train[c_append], trees.x_dict, c_append)
                c_append += 1
                # we don't have a parse:
                yield inst

        if self.use_wordrep_tree:
            if file_parsed is None or files_parsed_path is None:
                sys.exit("Missing parsed file.")

        contents = open(file, encoding="ascii")

        nr_sent = 0
        instances = []
        ex_x = []
        ex_y = []
        include_ex_z = (self.wordrep_dict is not None and not self.use_wordrep_tree)
        if include_ex_z:
            ex_z = []

        for line in contents:
            if "-DOCSTART" in line:
                continue
            toks = line.split("\t")
            if len(toks) < 9:
                if max_sent_len > len(ex_x) > 0:  #len(ex_x) > 1 # escape one-word sentences
                    nr_sent += 1
                    instances.append([ex_x, ex_y, ex_z] if include_ex_z else [ex_x, ex_y])
                if nr_sent >= max_nr_sent:
                    break
                ex_x = []
                ex_y = []
                if include_ex_z:
                    ex_z = []
            else:
                tag = toks[0]
                word = toks[5]
                if word not in self.word_dict:
                    self.word_dict.add(word)
                if tag not in self.tag_dict:
                    self.tag_dict.add(tag)
                ex_x.append(word)
                ex_y.append(tag)
                if include_ex_z:
                    # chain wordrep
                    ex_z.append(self.normalize_word(word))

        # add parsed data to use tree wordreps
        if self.use_wordrep_tree:
            for c, instance in enumerate(get_tree(len(instances))):
                #get parsed data
                instances[c].append(instance)

        return instances

    def write_conll_instances(self, gold, predictions, file, sep=" ", is_muc=False):
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

