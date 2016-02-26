import logging

import numpy as np

import eval.ner.sequences.discriminative_sequence_classifier as dsc


class StructuredPerceptron(dsc.DiscriminativeSequenceClassifier):
    """ Implements a first order CRF"""

    def __init__(self, observation_labels, state_labels, feature_mapper,
                 num_epochs=10, learning_rate=1.0, averaged=True):
        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        #self.params_per_epoch = []
        self.params_per_epoch = np.zeros(self.feature_mapper.get_num_features())
        self.loaded_model = None  # using existing model parameters

    def train_supervised(self, dataset, devset=None):
        logger = logging.getLogger(__name__)
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        num_examples = dataset.size()
        for epoch in range(self.num_epochs):
            num_labels_total = 0
            num_mistakes_total = 0
            for i in range(num_examples):
                sequence = dataset.seq_list[i]
                num_labels, num_mistakes = self.perceptron_update(sequence)
                num_labels_total += num_labels
                num_mistakes_total += num_mistakes
            #self.params_per_epoch.append(self.parameters.copy())
            self.params_per_epoch += self.parameters
            if devset:
                acc_dev = self.get_dev_accuracy(devset)
                logger.info("Epoch: {} Devset accuracy: {}".format(epoch, acc_dev))
                # Accuracy on train set
                #acc = 1.0 - num_mistakes_total/num_labels_total
                #print("Epoch: {} Accuracy: {}".format(epoch, acc))
        self.trained = True

        if self.averaged:
            #new_w = 0
            #for old_w in self.params_per_epoch:
            #    new_w += old_w
            #new_w = new_w / len(self.params_per_epoch)
            #self.parameters = new_w
            self.parameters = self.params_per_epoch / self.num_epochs

    def perceptron_update(self, sequence):
        num_labels = 0
        num_mistakes = 0

        predicted_sequence, _ = self.viterbi_decode(sequence)

        y_hat = predicted_sequence.y

        # Update initial features.
        y_t_true = sequence.y[0]
        y_t_hat = y_hat[0]

        if y_t_true != y_t_hat:
            true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true)
            self.parameters[true_initial_features] += self.learning_rate
            hat_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_hat)
            self.parameters[hat_initial_features] -= self.learning_rate

        for pos in range(len(sequence.x)):
            y_t_true = sequence.y[pos]
            y_t_hat = y_hat[pos]

            # Update emission features.
            num_labels += 1
            if y_t_true != y_t_hat:
                num_mistakes += 1
                #emission_features can be real-valued
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                assert len(list(true_emission_features.keys())) == len(set(true_emission_features.keys()))
                self.parameters[list(true_emission_features.keys())] += self.learning_rate * np.array(
                    list(true_emission_features.values()))
                hat_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
                self.parameters[list(hat_emission_features.keys())] -= self.learning_rate * np.array(
                    list(hat_emission_features.values()))
            if pos > 0:
                ## update bigram features
                ## If true bigram != predicted bigram update bigram features
                prev_y_t_true = sequence.y[pos - 1]
                prev_y_t_hat = y_hat[pos - 1]
                if y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat:
                    true_transition_features = self.feature_mapper.get_transition_features(sequence, pos - 1, y_t_true,
                                                                                           prev_y_t_true)
                    self.parameters[true_transition_features] += self.learning_rate
                    hat_transition_features = self.feature_mapper.get_transition_features(sequence, pos - 1, y_t_hat,
                                                                                          prev_y_t_hat)
                    self.parameters[hat_transition_features] -= self.learning_rate

            """
            # trigram feat
            if pos > 1:
                prev_prev_y_t_true = sequence.y[pos-2]
                prev_prev_y_t_hat = y_hat[pos-2]
                prev_y_t_true = sequence.y[pos-1]
                prev_y_t_hat = y_hat[pos-1]
                if y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat or prev_prev_y_t_true != prev_prev_y_t_hat:
                    true_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_true, prev_y_t_true, prev_prev_y_t_true)
                    self.parameters[true_transition_features] += self.learning_rate
                    hat_transition_features = self.feature_mapper.get_transition_features(sequence, pos-1, y_t_hat, prev_y_t_hat, prev_prev_y_t_hat)
                    self.parameters[hat_transition_features] -= self.learning_rate
            """
        # Update final features.
        pos = len(sequence.x)
        y_t_true = sequence.y[pos - 1]
        y_t_hat = y_hat[pos - 1]

        if y_t_true != y_t_hat:
            true_final_features = self.feature_mapper.get_final_features(sequence, y_t_true)
            self.parameters[true_final_features] += self.learning_rate
            hat_final_features = self.feature_mapper.get_final_features(sequence, y_t_hat)
            self.parameters[hat_final_features] -= self.learning_rate

        return num_labels, num_mistakes

    def get_dev_accuracy(self, devset):
        pred_dev = self.viterbi_decode_corpus(devset)
        eval_dev = self.evaluate_corpus(devset, pred_dev)
        return eval_dev

    def save_model_numpy(self, dirname):
        np.save(dirname + "/parameters", self.parameters)

    def load_model_numpy(self, filename):
        self.parameters = np.load(filename)
        self.loaded_model = filename
        self.num_epochs = None

    def save_model(self, dirname):
        fn = open(dirname + "/parameters.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("{}\t{}\n".format(p_id, p))
        fn.close()

    def load_model(self, filename):
        fn = open(filename)
        for line in fn:
            toks = line.strip().split("\t")
            assert len(toks) == 2
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
        self.loaded_model = filename
        self.num_epochs = None
