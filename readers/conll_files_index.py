from os import walk

__author__ = 'sim'


class ConllFilesIndex:
    def __init__(self, files_path):
        """
        Obtain a mapping between successfully parsed files (*.tab files with indices) and its position in the
         concatenated conll file (output.conll).

        :param files_path: path containing *.tab (conll) files obtained with alpino2conll
        """
        self.files_path = files_path
        self.fileids = set()

    def create_mapping(self):
        fileid_to_position = {}
        for _, _, filenames in walk(self.files_path):
            ids = sorted([eval(f.split(".")[0]) for f in filenames if f.endswith(".tab")])
        for i in ids:
            fileid_to_position[i] = len(fileid_to_position) + 1

        self.fileid_to_position = fileid_to_position

        position_to_fileid = {v: k for k, v in fileid_to_position.items()}
        self.position_to_fileid = position_to_fileid

    def create_ids_set(self):
        for _, _, filenames in walk(self.files_path):
            fileids = {eval(f.split(".")[0]) for f in filenames if f.endswith(".tab")}
        self.fileids = fileids