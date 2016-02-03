import numpy as np
# from scipy import sparse

from util.log_domain import sselogsum, logzero

""" 
Container for HMM inference methods.
"""


def project_kbest(v, k_prop=1 / 8):
    """
    Only keep k largest coefficients; remaining elements are set to logzero. Form of regularization.
    Following Grave et al. 2013 (for a 128-state model they set k=16).
    So k proportion should be roughly 1/8 of the state size.

    TODO: use different data structure
    (sparse array) to bring speed again.
    :param v: vector
    :param k_prop: k proportion of states to keep
    """
    assert isinstance(v, np.ndarray)
    k = k_prop * v.shape[0]
    k_largest = v.argsort()[-k:]
    v_approx = np.zeros(v.shape[0], 'f') + logzero()
    v_approx[k_largest] = v[k_largest]

    return rescale_projected(v_approx, sselogsum(v))


def rescale_projected(v, total):
    return v + (total - sselogsum(v))


class HMMInference:
    """ Forward backward """

    def __init__(self, approximate=False):
        # log sum table: table and lookup methods for efficient logsums
        self.approximate = approximate

    def compute_fb(self, initial_scores, transition_scores, final_scores, emission_scores):
        length, N = emission_scores.shape
        ll, forward = self.run_forward(initial_scores,
                                       transition_scores,
                                       final_scores,
                                       emission_scores,
                                       length,
                                       N)

        ll2, backward = self.run_backward(initial_scores,
                                          transition_scores,
                                          final_scores,
                                          emission_scores,
                                          length,
                                          N)
        # assert np.isclose(ll, ll2)
        #print("Forward trellis: {}\nLL: {}".format(forward, ll))
        #print("Backward trellis: {}\nLL: {}".format(backward, ll2))
        return forward, ll, backward, ll2

    # @profile
    def compute_posteriors(self, initial_scores, transition_scores, final_scores, emission_scores):
        """ 
        Compute state (gamma) and transition (di-gamma)
        posteriors.
        State posterior is the probability of each state at each 
         time/position given the observation sequence.
        Transition posterior is the joint probability of being in two 
         states at two consecutive times/positions given the observation
         sequence.

        To compute the two posteriors, forward (alpha) and backward (beta)
         trellises must be "filled", which is done through forward-backward
         algorithm.

        """
        forward, ll, backward, ll2 = self.compute_fb(initial_scores, transition_scores, final_scores, emission_scores)
        #assert np.isclose(ll, ll2)
        length, N = emission_scores.shape
        #print("Forward trellis: {}\nLL: {}".format(forward, ll))
        #print("Backward trellis: {}\nLL: {}".format(backward, ll2))

        #Multiply forward and backward matrices and normalize by ll to obtain state posteriors.
        state_posteriors = (forward + backward) - ll
        #print("state posteriors FB: {}".format(state_posteriors))
        #assert state_posteriors.shape == (length, N)
        #Obtain transition posteriors based on forward, backward, transition_scores and emission_scores matrices
        #a try in vectorization removing loops over current and previous state:
        transition_posteriors = np.zeros([length - 1, N, N], 'f')
        for pos in range(length - 1):
            transition_posteriors[pos, :, :] = forward[pos, :] + \
                                               transition_scores[pos, :, :] + \
                                               emission_scores[pos + 1, :].reshape(-1, 1) + \
                                               backward[pos + 1, :].reshape(-1, 1)
            transition_posteriors[pos, :, :] -= ll

        #print("trans.posteriors: {}".format(transition_posteriors))
        #de-log:
        state_posteriors = np.exp(state_posteriors)
        transition_posteriors = np.exp(transition_posteriors)

        return state_posteriors, transition_posteriors, ll

    def compute_state_posteriors(self, initial_scores, transition_scores, final_scores, emission_scores):
        """
        Compute state posteriors.
        State posterior is the probability of each state at each
         time/position given the observation sequence.
        To compute the posteriors, forward (alpha) and backward (beta)
         trellises must be "filled", which is done through forward-backward
         algorithm.

        """
        forward, ll, backward, ll2 = self.compute_fb(initial_scores, transition_scores, final_scores, emission_scores)
        #assert np.isclose(ll, ll2)
        #print("Forward trellis: {}\nLL: {}".format(forward, ll))
        #print("Backward trellis: {}\nLL: {}".format(backward, ll2))
        #Multiply forward and backward matrices and normalize by ll to obtain state posteriors.
        return (forward + backward) - ll

    # @profile
    def run_forward(self, initial_scores, transition_scores, final_scores, emission_scores, length, N):
        """ Forward trellis scores."""
        #try an alternative vectorized implementation  of the loops here that eliminates the loop over "current_state"...
        forward = np.zeros([length, N], 'f') + logzero()

        #Initialization
        forward[0, :] = emission_scores[0, :] + initial_scores

        #Forward loop
        for pos in range(1, length):
            #in log: sum the forward scores of previous position states and transition scores from previous state
            # up to the current state; logsum all this to obtain a scalar.
            #logsum over the rows
            forward_last = project_kbest(forward[pos - 1, :]) if self.approximate else forward[pos - 1, :]
            for current_state in range(N):
                forward[pos, current_state] = sselogsum(forward_last +
                                                        transition_scores[pos - 1, current_state, :])
            #add emission
            forward[pos, :] += emission_scores[pos, :]
        #Termination
        ll = sselogsum(project_kbest(forward[length - 1, :]) + final_scores) if self.approximate else \
            sselogsum(forward[length - 1, :] + final_scores)

        return ll, forward

    def run_backward(self, initial_scores, transition_scores, final_scores, emission_scores, length, N):
        """ Backward trellis scores. """
        backward = np.zeros([length, N], 'f') + logzero()
        #Initialization
        backward[length - 1, :] = final_scores
        #Backward loop
        for pos in range(length - 2, -1, -1):
            #transition_oldbackward = transition_scores[pos, :, :] + backward[pos+1, :].reshape(-1, 1)
            #transition_oldbackward += emission_scores[pos+1, :].reshape(-1,1)
            #backward[pos, :] = np.apply_along_axis(logsum, 0, transition_oldbackward)
            product = project_kbest(backward[pos + 1, :] + emission_scores[pos + 1, :]) if self.approximate else \
                backward[pos + 1, :] + emission_scores[pos + 1, :]
            for current_state in range(N):
                backward[pos, current_state] = sselogsum(product + transition_scores[pos, :, current_state])

        ll = sselogsum(project_kbest(backward[0, :] + emission_scores[0, :]) + initial_scores) if self.approximate else \
            sselogsum(backward[0, :] + emission_scores[0, :] + initial_scores)

        return ll, backward

    def run_viterbi(self, initial_scores, transition_scores, final_scores, emission_scores, length, N):

        # Variables storing the Viterbi scores.
        viterbi_scores = np.zeros([length, N], 'f') + logzero()
        # Variables storing the paths to backtrack.
        viterbi_paths = -np.ones([length, N], dtype=int)
        # Most likely sequence.
        best_path = -np.ones(length, dtype=int)

        # Initialization.
        viterbi_scores[0, :] = emission_scores[0, :] + initial_scores

        # Viterbi loop.
        for pos in range(1, length):
            for current_state in range(N):
                viterbi_scores[pos, current_state] = np.max(viterbi_scores[pos - 1, :] +
                                                            transition_scores[pos - 1, current_state, :])
                viterbi_scores[pos, current_state] += emission_scores[pos, current_state]
                viterbi_paths[pos, current_state] = np.argmax(viterbi_scores[pos - 1, :] +
                                                              transition_scores[pos - 1, current_state, :])
        # Termination.
        best_score = np.max(viterbi_scores[length - 1, :] + final_scores)
        best_path[length - 1] = np.argmax(viterbi_scores[length - 1, :] + final_scores)

        # Backtrack.
        for pos in range(length - 2, -1, -1):
            best_path[pos] = viterbi_paths[pos + 1, best_path[pos + 1]]

        return best_path, best_score