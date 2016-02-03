import os

import numpy as np


def chunk_seqs_by_size(l, size, permute=False):
    """
    Split a list into chunks.
    Optionally permute the chunks.
    
    :param l: list
    :param size: chunk size
    :param permute:
    """
    splist = list(range(0, len(l), size))
    if permute:
        np.random.shuffle(splist)
    for i in splist:
        yield l[i:i + size]


def chunk_seqs_by_size_gen(g, size):
    """
    Yield "size"number of instances.

    :param g: list or generator
    :param size: chunk size
    """

    l = []
    for c, i in enumerate(g, 1):
        l.append(i)
        if c % size == 0:
            yield l
            l = []
    if l:
        yield l


def chunk_seqs(l, n):
    """
    Split a list into n chunks.

    :param l: list
    :param n: desired number of chunks
    """
    size = (len(l) // n) + 1
    for i in range(0, len(l), size):
        yield l[i:i + size]


def line_reader(f, skip=0):
    with open(f) as in_f:
        for c, l in enumerate(in_f, 1):
            if c <= skip:
                continue
            yield l


def nparr_to_str(nparr):
    assert len(nparr.shape) == 1, "Can only process 1-dimensional arrays: shape {}".format(nparr.shape)
    return " ".join(nparr.astype(np.str))


def npmultiarr_to_str(npmultiarr):
    def twodimarr_to_str(nparr):
        return (nparr_to_str(l) for l in nparr)

    if len(npmultiarr.shape) == 2:
        return twodimarr_to_str(npmultiarr)
    elif len(npmultiarr.shape) == 3:
        for i in range(npmultiarr.shape[2]):
            yield i, twodimarr_to_str(npmultiarr[:, :, i])


def getFileList(topdir, identifiers=None, all_levels=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    filelist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for filename in files:
            get = False
            for i in identifiers:
                if i in filename:
                    get = True
            if get:
                fullname = os.path.join(root, filename)
                filelist.append(fullname)

    return filelist