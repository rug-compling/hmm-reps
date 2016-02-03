import sys

from sequences.sequence import Sequence


__author__ = 'sim'


class SequenceLabel(Sequence):
    def __init__(self, sequence_list, x, y, nr, z=None, w=None, t=None):
        """
        :param z: normalized (unk) sequence
        :param w: wordrep sequence
        :param t: normalized tree suitable for tree wordrep
        """
        super().__init__(sequence_list, x, nr)
        if z is not None and t is not None:
            sys.exit("Both chain and tree rep provided.")
        self.y = y
        self.z = z  # x as used for wordrep
        self.w = w  # labels/states of z (in wordreps)
        self.t = t
        self.u = None

    def __str__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            if self.z is not None:
                zi = self.z[i]
                if self.w is not None:
                    wi = self.w[i]
                    rep += "{}/{}-{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                                 self.sequence_list.y_dict.get_label_name(yi),
                                                 self.sequence_list.wordrep_dict.get_label_name(zi),
                                                 wi)
                else:
                    rep += "{}/{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                              self.sequence_list.y_dict.get_label_name(yi),
                                              self.sequence_list.wordrep_dict.get_label_name(zi))
            elif self.t is not None:
                ti = self.t[i + 1].get_name()
                if self.t[i + 1].max_state is not None:
                    rep += "{}/{}-{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                                 self.sequence_list.y_dict.get_label_name(yi),
                                                 self.sequence_list.wordrep_dict.get_label_name(ti),
                                                 self.t[i + 1].max_state)
                else:
                    rep += "{}/{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                              self.sequence_list.y_dict.get_label_name(yi),
                                              self.sequence_list.wordrep_dict.get_label_name(ti))
            else:
                rep += "{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                       self.sequence_list.y_dict.get_label_name(yi))
        return rep

    def __repr__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            yi = self.y[i]
            if self.z is not None:
                zi = self.z[i]
                if self.w is not None:
                    wi = self.w[i]
                    rep += "{}/{}-{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                                 self.sequence_list.y_dict.get_label_name(yi),
                                                 self.sequence_list.wordrep_dict.get_label_name(zi),
                                                 wi)
                else:
                    rep += "{}/{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                              self.sequence_list.y_dict.get_label_name(yi),
                                              self.sequence_list.wordrep_dict.get_label_name(zi))
            elif self.t is not None:
                ti = self.t[i + 1].get_name()
                if self.t[i + 1].max_state is not None:
                    rep += "{}/{}-{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                                 self.sequence_list.y_dict.get_label_name(yi),
                                                 self.sequence_list.wordrep_dict.get_label_name(ti),
                                                 self.t[i + 1].max_state)
                else:
                    rep += "{}/{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                              self.sequence_list.y_dict.get_label_name(yi),
                                              self.sequence_list.wordrep_dict.get_label_name(ti))
            else:
                rep += "{}/{} ".format(self.sequence_list.x_dict.get_label_name(xi),
                                       self.sequence_list.y_dict.get_label_name(yi))
        return rep

    def copy_sequence(self):
        if self.z is not None:
            if self.w is not None:
                s = SequenceLabel(self.sequence_list, self.x[:], self.y[:], self.nr, self.z[:], self.w[:])
            else:
                s = SequenceLabel(self.sequence_list, self.x[:], self.y[:], self.nr, self.z[:])
        elif self.t is not None:
            s = SequenceLabel(self.sequence_list, self.x[:], self.y[:], self.nr, None, None, self.t.shallow_copy())
        else:
            s = SequenceLabel(self.sequence_list, self.x[:], self.y[:], self.nr)
        return s

    def update_from_sequence(self, new_y):
        '''Returns a new sequence equal to the previous but with y set to newy'''
        if self.z is not None:
            if self.w is not None:
                s = SequenceLabel(self.sequence_list, self.x, new_y, self.nr, self.z[:], self.w[:])
            else:
                s = SequenceLabel(self.sequence_list, self.x, new_y, self.nr, self.z[:])
        elif self.t is not None:
            s = SequenceLabel(self.sequence_list, self.x, new_y, self.nr, None, None, self.t.shallow_copy())
        else:
            s = SequenceLabel(self.sequence_list, self.x, new_y, self.nr)
        return s