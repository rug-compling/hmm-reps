"""
Prepare BLLIP 1987-89 WSJ Corpus Release 1:

- Remove # tags which cause errors in LTH pennconverter.
- Remove PTB sents from BLLIP.
- Prepare for parsing: convert tagged text in Brown format (word/pos)
to conll 8-column (for MST parser)
"""

import os
import sys

from util.util import getFileList


def remove_hash(path):
    """ for all BLLIP files, substitute # with -

    To speed this up, we concatenate small files per year, e.g.:
    #find `pwd` -iname '*.*' |xargs cat > bllip_87_89_wsj/1987/1987.all

    :param path: dir path to concatenated BLLIP files: bllip_87_89_wsj/
    """

    print("Obtaining files...")
    files = [f for f in getFileList(path, all_levels=True) if
             "1987.all" in f or
             "1988.all" in f or
             "1989.all" in f]
    assert len(files) == 3

    for f in files:
        print(f)
        outfile = "{0}.nohash".format(f)
        try:
            OUT = open(outfile, "w")
        except FileNotFoundError:
            os.makedirs(os.path.dirname(outfile))
            OUT = open(outfile, "w")

        with open(f) as IN:
            try:
                text = IN.read()
            except UnicodeDecodeError:
                continue
            OUT.write(text.replace("#", "-"))


def remove_bllip_ptb(bllip_file, ptb_file):
    """
    Remove sents in bllip which occur in PTB (all sections).
    Based on lexical overlap.

    :param bllip_file: one sent per line; only year 1989 (PTB is 1989): bllip_87_89_wsj/1989/1989.sents
    :param ptb_file: PTB_conll_NPpatched/ptb00_24.sents
    """
    bllip = open(bllip_file).readlines()
    ptb = set(open(ptb_file).readlines())  # set for O(1) lookup

    bllip_corr = [b for b in bllip if b not in ptb]
    diff_n = len(bllip) - len(bllip_corr)

    print("{} sents excluded".format(diff_n))

    with open("{0}.noptb".format(bllip_file), "w") as OUT:
        for i in bllip_corr:
            OUT.write(i)


def brown2conll_print(col=8):
    """
    Based on a tagged file in Brown format (slash-sep'd words and pos),
    produce Conll format.
    Instead of lemma and generalized POS, we emit dummy values which
    are either word or POS.

    argv: tagged_file : bllip_87_89_wsj/bllip.noptb.tagged
    :param col : n of columns, choosing 10 adds two underscores : default 8
    > bllip.noptb.tagged.conll
    """

    with open(sys.argv[1]) as IN:
        lines = IN.readlines()
        for line in lines:
            idx = 0
            if not line.strip():
                continue
            for i in line.split():
                idx += 1
                try:
                    w, t = i.rsplit("/", 1)
                except ValueError:
                    continue
                print("{0}\t{1}\t{1}\t{2}\t{2}\t_\t0\tLAB".format(idx, w, t))
            print()


def brown2conll(f, col=8, ext=".conll"):
    """
    Based on a tagged file in Brown format (slash-sep'd words and pos),
    produce Conll format.
    Instead of lemma and generalized POS, we emit dummy values which
    are either word or POS.

    argv: tagged_file : bllip_87_89_wsj/bllip.noptb.tagged
    :param col: n of columns, choosing 10 adds two underscores : default 8
    """

    with open(f) as IN, open("{}{}".format(f, ext), "w") as OUT:
        lines = IN.readlines()
        for line in lines:
            idx = 0
            if not line.strip():
                continue
            for i in line.split():
                idx += 1
                try:
                    w, t = i.rsplit("/", 1)
                except ValueError:
                    continue
                OUT.write("{0}\t{1}\t{1}\t{2}\t{2}\t_\t0\tLAB\n".format(idx, w, t))
            OUT.write("\n")


if __name__ == "__main__":
    remove_bllip_ptb(sys.argv[1], sys.argv[2])