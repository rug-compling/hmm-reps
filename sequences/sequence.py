class Sequence(object):
    def __init__(self, sequence_list, x, nr):
        self.x = x
        self.nr = nr
        self.sequence_list = sequence_list

    def size(self):
        '''Returns the size of the sequence.'''
        return len(self.x)

    def __len__(self):
        return len(self.x)

    def copy_sequence(self):
        '''Performs a deep copy of the sequence'''
        s = Sequence(self.sequence_list, self.x[:], self.nr)
        return s

    def __str__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            rep += "{} ".format(self.sequence_list.x_dict.get_label_name(xi))
        return rep

    def __repr__(self):
        rep = ""
        for i, xi in enumerate(self.x):
            rep += "{} ".format(self.sequence_list.x_dict.get_label_name(xi))
        return rep


