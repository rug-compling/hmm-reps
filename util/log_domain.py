import math
import sselogsumexp

import numpy as np


# ###########################################################################
# Functions to compute in log-domain.
# ###########################################################################

def logzero():
    return -np.inf


def safe_log(x):
    if x == 0:
        return logzero()
    return np.log(x)


def logsum(v):
    '''
    Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.
    Math.log is faster than np.log.
    '''
    c = np.max(v)
    return c + math.log(np.sum(np.exp(v - c)))


############################################################################

def sselogsum(v):
    """
    Fastest log sum exp on v array:
    https://github.com/rmcgibbo/logsumexp

    :param v: must be np.type32
    """
    return sselogsumexp.logsumexp(v)


class LogSumTable:
    def __init__(self):
        self.logsum_table = self.init_logsum_table()

    def logsum_pair_table_interp(self, diff):
        """
        Return the log1p term from precomputed table by interpolation.
        Cf. Treba

        Minimax log sum approximation might be even faster and more precise, TODO

        :param diff: x-y or y-x
        """

        index = -int(diff)
        w = -diff - index
        val1 = self.logsum_table[index]
        val2 = self.logsum_table[index + 1]

        return val1 + (w * (val2 - val1))

    def logsum_pair_table(self, diff):
        """
        No interpolation.
        """
        return self.logsum_table[-int(diff)]

    def logsum(self, v):
        '''
        Return log(v[0]+v[1]+...), avoiding arithmetic underflow/overflow.

        '''
        res = logzero()
        for val in v:
            res = self.logsum_pair(res, val)
        return res

    def logsum_pair(self, x, y):
        """


        :param x: log(x)
        :param y: log(y)

        """
        if x == logzero():
            return y
        #if y == logzero():
        #    return x

        if y > x:
            temp = x
            x = y
            y = temp

        neg_diff = y - x

        #if neg_diff <= -54:
        #    return x

        result = self.logsum_pair_table(neg_diff)

        return x + result

    def init_logsum_table(self, min_val=-54):
        """
        Return a logsum table to use for fast lookup.
        Compute as log1p(exp(x))
        x
        -54 is the lowest value needed as shown by
        M.Hulden: Treba, Efficient Numerically Stable EM for PFA
        http://jmlr.org/proceedings/papers/v21/hulden12a/hulden12a.pdf
        """
        logsum_table = np.zeros((-min_val) + 1)
        for i in range(len(logsum_table)):
            logsum_table[i] = np.log1p(np.exp(-i))

        return logsum_table
