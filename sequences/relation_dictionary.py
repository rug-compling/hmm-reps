__author__ = 'sim'


class RelationDictionary(dict):
    def __init__(self, label_names=None):
        if not label_names: label_names = []
        self.ids_to_names = {}
        for name in label_names:
            self.add(name)

    def __len__(self):
        return len(self.ids_to_names)

    def add(self, name):
        if name in self:
            return
        label_id = len(self.ids_to_names)
        self[name] = label_id
        if label_id not in self.ids_to_names:
            # self.ids_to_names[label_id] = []
            self.ids_to_names[label_id] = name

    def add_fixed_id(self, names, label_id):
        for name in names:
            self[name] = label_id
            # self.ids_to_names[label_id].append(name)

    def get_label_name(self, label_id):
        return self.ids_to_names[label_id]

    def get_label_id(self, name):
        return self[name]

    def write(self, path):
        import pickle

        pickle.dump(self, open(path, "wb"))