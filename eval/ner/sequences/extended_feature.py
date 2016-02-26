import sys
import numpy as np


#######################
#### Feature Class
### Extracts features from a labeled corpus
#######################
from eval.ner.readers.brown import prepare_cluster_map
from eval.ner.sequences.id_feature import IDFeatures


class ExtendedFeatures(IDFeatures):
    def __init__(self, dataset, brown_cluster_file=None):
        super().__init__(dataset)
        self.brown_cluster_file = brown_cluster_file
        self.w_to_clusterid = None
        if brown_cluster_file:
            self.w_to_clusterid = prepare_cluster_map(self.brown_cluster_file)
            assert self.w_to_clusterid is not None

        # use emission features:
        self.alphanumeric = False
        self.alldigits = False
        self.brown_id = False
        self.brown_id_plus1 = False
        self.brown_id_plus2 = False
        self.brown_id_minus1 = False
        self.brown_id_minus2 = False
        self.brown_prefix = False  # prefix length features; same for all brown_id
        self.brown_prefix_lengths = []
        self.capitalized = False
        self.cappattern = False
        self.hyphen = False
        self.id = False
        self.id_plus1 = False
        self.id_plus2 = False
        self.id_minus1 = False
        self.id_minus2 = False
        self.prefix = False
        self.rep_id = False
        self.rep_id_plus1 = False
        self.rep_id_plus2 = False
        self.rep_id_minus1 = False
        self.rep_id_minus2 = False
        self.suffix = False
        self.uppercased = False

    def set_baseline_features(self):
        """
        Use listed emission features as baseline
        """
        self.capitalized = True
        self.cappattern = True
        self.hyphen = True
        self.id = True
        self.id_plus1 = True
        self.id_plus2 = True
        self.id_minus1 = True
        self.id_minus2 = True
        self.prefix = True
        self.suffix = True
        self.uppercased = True

    def get_emission_features(self, sequence, pos, y):
        """
        Handles previous emissions by expanding the feature_cache dictionaries
        """
        # w
        x = sequence.x[pos]
        # w-1
        if pos > 0:
            x_min1 = sequence.x[pos-1]
        else:
            x_min1 = -1
        # w-2
        if pos > 1:
            x_min2 = sequence.x[pos-2]
        else:
            x_min2 = -2
        # w+1
        if pos < len(sequence.x)-1:
            x_plus1 = sequence.x[pos+1]
        else:
            x_plus1 = -3
        # w+2
        if pos < len(sequence.x)-2:
            x_plus2 = sequence.x[pos+2]
        else:
            x_plus2 = -4

        if x not in self.node_feature_cache:
            self.node_feature_cache[x] = {}
        if x_min1 not in self.node_feature_cache[x]:
            self.node_feature_cache[x][x_min1] = {}
        if x_min2 not in self.node_feature_cache[x][x_min1]:
            self.node_feature_cache[x][x_min1][x_min2] = {}
        if x_plus1 not in self.node_feature_cache[x][x_min1][x_min2]:
            self.node_feature_cache[x][x_min1][x_min2][x_plus1] = {}
        if x_plus2 not in self.node_feature_cache[x][x_min1][x_min2][x_plus1]:
            self.node_feature_cache[x][x_min1][x_min2][x_plus1][x_plus2] = {}

        if y not in self.node_feature_cache[x][x_min1][x_min2][x_plus1][x_plus2]:
            node_idx = {}
            node_idx = self.add_emission_features(sequence, pos, y, node_idx)
            self.node_feature_cache[x][x_min1][x_min2][x_plus1][x_plus2][y] = node_idx
        idx = self.node_feature_cache[x][x_min1][x_min2][x_plus1][x_plus2][y]

        return idx

    def add_emission_features(self, sequence, pos, y, features):
        # w
        x = sequence.x[pos]
        # w-1
        if pos > 0:
            x_min1 = sequence.x[pos-1]
        else:
            x_min1 = -1
        # w-2
        if pos > 1:
            x_min2 = sequence.x[pos-2]
        else:
            x_min2 = -2
        # w+1
        if pos < len(sequence.x)-1:
            x_plus1 = sequence.x[pos+1]
        else:
            x_plus1 = -3
        # w+2
        if pos < len(sequence.x)-2:
            x_plus2 = sequence.x[pos+2]
        else:
            x_plus2 = -4

        # Get tag name from ID.
        y_name = self.dataset.y_dict.get_label_name(y)
        # Get word name from ID.
        x_name = self.dataset.x_dict.get_label_name(x)
        word = str(x_name)
        # w-1
        if x_min1 == -1:  # if no previous word
            word_min1 = "out-of-seq-left"
        else:
            x_min1_name = self.dataset.x_dict.get_label_name(x_min1)
            word_min1 = str(x_min1_name)
        # w-2
        if x_min2 == -2:  # if no pre-previous word
            word_min2 = "out-of-seq-left"
        else:
            x_min2_name = self.dataset.x_dict.get_label_name(x_min2)
            word_min2 = str(x_min2_name)
        # w+1
        if x_plus1 == -3:  # if no next word
            word_plus1 = "out-of-seq-right"
        else:
            x_plus1_name = self.dataset.x_dict.get_label_name(x_plus1)
            word_plus1 = str(x_plus1_name)
        # w+3
        if x_plus2 == -4:  # if no post-next word
            word_plus2 = "out-of-seq-right"
        else:
            x_plus2_name = self.dataset.x_dict.get_label_name(x_plus2)
            word_plus2 = str(x_plus2_name)

        if self.id:
            # Generate feature name.
            feat_name = "id:{}::{}".format(word, y_name)
            self.features_used.add("id")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.id_minus1:
            feat_name = "id-1:{}::{}".format(word_min1, y_name)
            self.features_used.add("id-1")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.id_minus2:
            feat_name = "id-2:{}::{}".format(word_min2, y_name)
            self.features_used.add("id-2")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.id_plus1:
            feat_name = "id+1:{}::{}".format(word_plus1, y_name)
            self.features_used.add("id+1")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.id_plus2:
            feat_name = "id+2:{}::{}".format(word_plus2, y_name)
            self.features_used.add("id+2")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.capitalized:
            # Iscapitalized
            if word.istitle():
                # Generate feature name.
                feat_name = "capitalized::{}".format(y_name)
                self.features_used.add("capitalized")
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features[feat_id] = 1

        if self.uppercased:
            # Allcapitalized
            if word.isupper():
                # Generate feature name.
                feat_name = "uppercased::{}".format(y_name)
                self.features_used.add("uppercased")
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features[feat_id] = 1

        if self.cappattern:
            # Capitalization pattern in window
            pattern = []
            # w-2
            if word_min2 == "out-of-seq-left":
                pattern.append("-")
            else:
                if word_min2.istitle():
                    pattern.append("C")
                else:
                    pattern.append("N")
            # w-1
            if word_min1 == "out-of-seq-left":
                pattern.append("-")
            else:
                if word_min1.istitle():
                    pattern.append("C")
                else:
                    pattern.append("N")
            # w
            if word.istitle():
                pattern.append("C")
            else:
                pattern.append("N")
            # w+1
            if word_plus1 == "out-of-seq-right":
                pattern.append("-")
            else:
                if word_plus1.istitle():
                    pattern.append("C")
                else:
                    pattern.append("N")
            # w+2
            if word_plus2 == "out-of-seq-right":
                pattern.append("-")
            else:
                if word_plus2.istitle():
                    pattern.append("C")
                else:
                    pattern.append("N")
            # Generate feature name.
            feat_name = "cappattern:{}::{}".format("".join(pattern), y_name)
            self.features_used.add("cappattern")
            # Get feature ID from name.
            feat_id = self.add_feature(feat_name)
            # Append feature.
            if feat_id != -1:
                features[feat_id] = 1

        if self.alldigits:
            # Alldigits
            if word.isdigit():
                # Generate feature name.
                feat_name = "number::{}".format(y_name)
                self.features_used.add("number")
               # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features[feat_id] = 1

        if self.alphanumeric:
            # Alphanumeric
            if word.isalnum():
                # Generate feature name.
                feat_name = "alphanumber::{}".format(y_name)
                self.features_used.add("alphanumber")
               # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features[feat_id] = 1

        if self.hyphen:
            # Hyphenized
            if "-" in word:
                # Generate feature name.
                feat_name = "hyphen::{}".format(y_name)
                self.features_used.add("hyphen")
                # Get feature ID from name.
                feat_id = self.add_feature(feat_name)
                # Append feature.
                if feat_id != -1:
                    features[feat_id] = 1

        if self.suffix:
            # Suffixes
            max_suffix = 3
            for i in range(max_suffix):
                if len(word) > i+1:
                    suffix = word[-(i+1):]
                    # Generate feature name.
                    feat_name = "suffix:{}::{}".format(suffix, y_name)
                    self.features_used.add("suffix")
                    # Get feature ID from name.
                    feat_id = self.add_feature(feat_name)
                    # Append feature.
                    if feat_id != -1:
                        features[feat_id] = 1

        if self.prefix:
            # Prefixes
            max_prefix = 3
            for i in range(max_prefix):
                if len(word) > i+1:
                    prefix = word[:i+1]
                    # Generate feature name.
                    feat_name = "prefix:{}::{}".format(prefix, y_name)
                    self.features_used.add("prefix")
                    # Get feature ID from name.
                    feat_id = self.add_feature(feat_name)
                    # Append feature.
                    if feat_id != -1:
                        features[feat_id] = 1

        # hmm wordrep features
        positions = []
        if self.rep_id:
            positions.append(0)
        if self.rep_id_minus1:
            if word_min1 != "out-of-seq-left":  # if no previous word
                positions.append(-1)
        if self.rep_id_minus2:
            if word_min2 != "out-of-seq-left":  # if no previous previous word
                positions.append(-2)
        if self.rep_id_plus1:
            if word_plus1 != "out-of-seq-right":  # if no next word
                positions.append(1)
        if self.rep_id_plus2:
            if word_plus2 != "out-of-seq-right":  # if no next next word
                positions.append(2)

        for position in positions:
            if sequence.w is not None:
                rep_id = sequence.w[pos+position]
                self.add_rep_feat(rep_id, position, y_name, features, is_tree=False)
            #changed sequence.t to sequence.u as seq.t is deleted
            if sequence.u is not None:
                try:
                    rep_id = sequence.u[pos+1+position] #offset of 1 because 0 is root
                except KeyError:
                    continue
                self.add_rep_feat(rep_id, position, y_name, features, is_tree=True)


        if self.brown_id:
            # w: Brown cluster id
            clusterid = None
            if word in self.w_to_clusterid:
                clusterid = self.w_to_clusterid[word]
            elif word.lower() in self.w_to_clusterid:
                clusterid = self.w_to_clusterid[word.lower()]
            if clusterid:
                feat_core_name = "brown_id"
                feat_name = "{}::{}::{}".format(feat_core_name, clusterid, y_name)
                self.features_used.add(feat_core_name)
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features[feat_id] = 1

                if self.brown_prefix:
                    features = self.add_brown_pref_feat(feat_core_name, clusterid, y_name, features, self.brown_prefix_lengths)

        # w window: Brown Cluster ids
        if self.brown_id_minus1:
            # wcluster-1
            if word_min1 != "out-of-seq-left":  # if no previous word
                clusterid = None
                if word_min1 in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_min1]
                elif word_min1.lower() in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_min1.lower()]
                if clusterid:
                    feat_core_name = "brown_id-1"
                    feat_name = "{}::{}::{}".format(feat_core_name, clusterid, y_name)
                    self.features_used.add(feat_core_name)
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                        features[feat_id] = 1

                    if self.brown_prefix:
                        features = self.add_brown_pref_feat(feat_core_name, clusterid, y_name, features, self.brown_prefix_lengths)

        if self.brown_id_minus2:
            # wcluster-2
            if word_min2 != "out-of-seq-left":  # if no pre-previous word
                clusterid = None
                if word_min2 in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_min2]
                elif word_min2.lower() in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_min2.lower()]
                if clusterid:
                    feat_core_name = "brown_id-2"
                    feat_name = "{}::{}::{}".format(feat_core_name, clusterid, y_name)
                    self.features_used.add(feat_core_name)
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                        features[feat_id] = 1

                    if self.brown_prefix:
                        features = self.add_brown_pref_feat(feat_core_name, clusterid, y_name, features, self.brown_prefix_lengths)

        if self.brown_id_plus1:
            # wcluster+1
            if word_plus1 != "out-of-seq-right":  # if no next word
                clusterid = None
                if word_plus1 in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_plus1]
                elif word_plus1.lower() in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_plus1.lower()]
                if clusterid:
                    feat_core_name = "brown_id+1"
                    feat_name = "{}::{}::{}".format(feat_core_name, clusterid, y_name)
                    self.features_used.add(feat_core_name)
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                         features[feat_id] = 1

                    if self.brown_prefix:
                        features = self.add_brown_pref_feat(feat_core_name, clusterid, y_name, features, self.brown_prefix_lengths)

        if self.brown_id_plus2:
            # wcluster+2
            if word_plus2 != "out-of-seq-right":  # if no next word
                clusterid = None
                if word_plus2 in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_plus2]
                elif word_plus2.lower() in self.w_to_clusterid:
                    clusterid = self.w_to_clusterid[word_plus2.lower()]
                if clusterid:
                    feat_core_name = "brown_id+2"
                    feat_name = "{}::{}::{}".format(feat_core_name, clusterid, y_name)
                    self.features_used.add(feat_core_name)
                    feat_id = self.add_feature(feat_name)
                    if feat_id != -1:
                        features[feat_id] = 1

                    if self.brown_prefix:
                        features = self.add_brown_pref_feat(feat_core_name, clusterid, y_name, features, self.brown_prefix_lengths)

        return features

    def add_brown_pref_feat(self, feat_core_name, clusterid, y_name, features, pref_lengths):
        """
        :param pref_lengths: list containing ints representing prefix lengths
        """
        for pref in pref_lengths:
            feat_name = "{}_p{}::{}::{}".format(feat_core_name, pref, clusterid[:pref], y_name)
            self.features_used.add("{}_p{}".format(feat_core_name, pref))
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features[feat_id] = 1

        return features

    # def get_transition_features(self, sequence, pos, y, y_prev, y_prev_prev):
    #     assert (pos >= 0 and pos < len(sequence.x))
    #
    #     if y not in self.edge_feature_cache:
    #         self.edge_feature_cache[y] = {}
    #     if y_prev not in self.edge_feature_cache[y]:
    #         self.edge_feature_cache[y][y_prev] = {}
    #     if y_prev_prev not in self.edge_feature_cache[y][y_prev]:
    #         edge_idx = []
    #         edge_idx = self.add_transition_features(sequence, pos, y, y_prev, y_prev_prev, edge_idx)
    #         self.edge_feature_cache[y][y_prev][y_prev_prev] = edge_idx
    #     idx = self.edge_feature_cache[y][y_prev][y_prev_prev]
    #
    #     return idx[:]

    # def add_transition_features(self, sequence, pos, y, y_prev, y_prev_prev, features):
    #     assert pos < len(sequence.x)-1
    #     # Get label name from ID.
    #     y_name = self.dataset.y_dict.get_label_name(y)
    #     # Get previous label names from ID.
    #     y_prev_name = self.dataset.y_dict.get_label_name(y_prev)
    #     y_prev_prev_name = self.dataset.y_dict.get_label_name(y_prev_prev)
    #     # Generate feature name.
    #     feat_name = "prev_tag:{}::{}".format(y_prev_name, y_name)
    #     self.features_used.add("prev_tag")
    #     # Get feature ID from name.
    #     feat_id = self.add_feature(feat_name)
    #     # Append feature.
    #     if feat_id != -1:
    #         features.append(feat_id)
    #
    #     feat_name = "prev_prev_tag:{}::{}".format(y_prev_prev_name, y_name)
    #     self.features_used.add("prev_prev_tag")
    #     # Get feature ID from name.
    #     feat_id = self.add_feature(feat_name)
    #     # Append feature.
    #     if feat_id != -1:
    #         features.append(feat_id)
    #
    #    return features


    def add_rep_feat(self, rep_id, position, y_name, features, is_tree=False):
        if position == 0:
            position = ""
        elif position == 1 or position == 2:
            position = "+{}".format(position)

        tree = "tree" if is_tree else ""
        # discrete
        if isinstance(rep_id, (int, np.int64, np.int32)):
            feat_name = "{}rep_id{}::{}::{}".format(tree, position, rep_id, y_name)
            self.features_used.add("{}rep_id{}".format(tree, position))
            feat_id = self.add_feature(feat_name)
            if feat_id != -1:
                features[feat_id] = 1
        # continuous
        elif isinstance(rep_id, np.ndarray):
            #max_i = np.max(rep_id)
            for c, i in enumerate(rep_id):
                #if i == max_i or i == max_i-1:
                #feat_name = "cont_{}rep_id{}::{}::{}::{}".format(tree, position, c, i, y_name)
                feat_name = "cont_{}rep_id{}::{}::{}".format(tree, position, c, y_name)
                self.features_used.add("cont_{}rep_id{}".format(tree, position))
                feat_id = self.add_feature(feat_name)
                if feat_id != -1:
                    features[feat_id] = i
        else:
            raise TypeError
            #sys.exit("unexpected type: {}".format(type(rep_id)))
