import logging

import numpy as np

from util.log_domain import logzero, sselogsum


__author__ = 'sim'


class HMMsplitmerge:
    # @profile
    def get_loss(self, N, forward, backward, ll, norm_emission_counts):
        """
        :param forward: forward trellis
        :param backward: backward trellis
        :param ll: log likelihood of the current sequence
        :param emission_scores: for access to state posteriors needed for weighting backward merges
        :param N: number of split states
        """
        logger = logging.getLogger(__name__)
        loss_seq = np.zeros(N / 2)  # there are N/2 possible merges
        assert forward.shape[1] == N

        n_merge = 0
        # iterate over possible merges
        for i in range(len(loss_seq)):
            i_merge = i + n_merge
            # prepare trellis
            #forward_merge = np.zeros((forward.shape[0], forward.shape[1]-1), 'f') + logzero()
            forward_to_merge = forward[:, i_merge:i_merge + 2]
            sum_split_forward = np.zeros((forward.shape[0], 1), 'f') + logzero()
            for row_n, row in enumerate(forward_to_merge):
                # sum split states
                sum_split_forward[row_n] = sselogsum(row)
            forward_merge = np.hstack((forward[:, :i_merge], sum_split_forward, forward[:, i_merge + 2:]))

            backward_to_merge = backward[:, i_merge:i_merge + 2]
            sum_split_backward = np.zeros((backward.shape[0], 1), 'f') + logzero()
            # incorporate weights for each element (Petrov 2009, p. 89)
            # weights are normalized emission counts
            # accumulated from state posteriors over all sequences
            assert backward.shape[1] == N
            backward_to_merge += norm_emission_counts[i_merge:i_merge + 2]
            for row_n, row in enumerate(backward_to_merge):
                # sum weighted split states
                sum_split_backward[row_n] = sselogsum(row)
            backward_merge = np.hstack((backward[:, :i_merge], sum_split_backward, backward[:, i_merge + 2:]))

            ll_merged_positions = np.zeros(forward_merge.shape[0], 'f') + logzero()
            fb_merge = forward_merge + backward_merge
            for row_n, row in enumerate(fb_merge):
                ll_merged_positions[row_n] = sselogsum(row)
            # likelihood for one merge at all t's
            #for row_n in range(forward_merge.shape[0]):
            #    ll_merged_positions[row_n] = sselogsum(forward_merge[row_n] + backward_merge[row_n])

            # get loss (difference in likelihoods)
            #assert np.all(ll_merged_positions < ll)
            #assert np.all((ll_merged_positions < ll).astype(int) |
            #              (np.isclose(ll_merged_positions, ll, rtol=0.01)).astype(int))
            loss_seq[i] = (ll_merged_positions - ll).sum()
            n_merge += 1

        return loss_seq