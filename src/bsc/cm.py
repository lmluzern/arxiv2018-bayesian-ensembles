import numpy as np
from scipy.special.basic import psi

from bsc.cv import VectorWorker


class ConfusionMatrixWorker(VectorWorker):
    # Worker model: Bayesianized Dawid and Skene confusion matrix ----------------------------------------------------------

    def _init_lnPi(self):
        # Returns the initial values for alpha and lnPi
        psi_alpha_sum = psi(np.sum(self.alpha0, 1))
        self.lnPi = psi(self.alpha0) - psi_alpha_sum[:, None, :]

        # init to prior
        self.alpha = np.copy(self.alpha0)


    def _calc_q_pi(self, alpha):
        '''
        Update the annotator models.
        '''
        psi_alpha_sum = psi(np.sum(alpha, 1))[:, None, :]
        return psi(self.alpha) - psi_alpha_sum


    def _post_alpha(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha.
        '''
        dims = self.alpha0.shape
        self.alpha = self.alpha0.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C == l + 1).T.dot(Tj)
                self.alpha[j, l, :] += counts


    def _post_alpha_data(self, E_t, C, doc_start, nscores):  # Posterior Hyperparameters
        '''
        Update alpha when C is the votes for one annotator, and each column contains a probability of a vote.
        '''
        dims = self.alpha0_data.shape
        self.alpha_data = self.alpha0_data.copy()

        for j in range(dims[0]):
            Tj = E_t[:, j]

            for l in range(dims[1]):
                counts = (C[:, l:l+1]).T.dot(Tj).reshape(-1)
                self.alpha_data[j, l, :] += counts


    def _read_lnPi(self, lnPi, l, C, Cprev, Krange, nscores, blanks=None):
        if l is None:
            if np.isscalar(Krange):
                Krange = np.array([Krange])[None, :]
            if np.isscalar(C):
                C = np.array([C])[:, None]

            result = lnPi[:, C, Krange]
            result[:, C == -1] = 0
        else:
            result = lnPi[l, C, Krange]
            if np.isscalar(C):
                if C == -1:
                    result = 0
            else:
                result[C == -1] = 0

        return result


    def _expand_alpha0(self, K, nscores):
        '''
        Take the alpha0 for one worker and expand.
        :return:
        '''
        L = self.alpha0.shape[0]

        # set priors
        if self.alpha0 is None:
            # dims: true_label[t], current_annoc[t],  previous_anno c[t-1], annotator k
            self.alpha0 = np.ones((L, nscores, K)) + 1.0 * np.eye(L)[:, :, None]
        else:
            self.alpha0 = self.alpha0[:, :, None]
            self.alpha0 = np.tile(self.alpha0, (1, 1, K))

        for midx in range(self.nModels):
            if self.alpha0_data[midx] is None:
                self.alpha0_data[midx] = np.ones((L, nscores, 1)) + 1.0 * np.eye(L)[:, :, None]
            elif self.alpha0_data[midx].ndim == 2:
                self.alpha0_data[midx] = self.alpha0_data[midx][:, :, None]


    def _calc_EPi(self, alpha):
        return alpha / np.sum(alpha, axis=1)[:, None, :]