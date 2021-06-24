from __future__ import division
import sys
import logging
import numpy as np
import cupy as cp
from scipy.stats import mode
from scipy.spatial.distance import squareform
from fastdtw import fastdtw

logging.getLogger().setLevel(logging.INFO)


class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN

    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function

    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """

    dtw_options = ["standard", "fastdtw"]

    def __init__(self, dtw, classes, n_neighbors, max_warping_window, subsample_step=1):
        self.classes = classes
        self.class_to_num = {value: index for index, value in enumerate(classes)}
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        assert dtw in self.dtw_options, f"Invalid dtw option, must be in {self.dtw_options}"
        self.dtw = dtw

    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer

        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """

        self.x = x
        self.l = l

    def _dtw_distance(self, ts_a, ts_b, d=lambda x, y: abs(x - y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """
        if self.dtw == "standard":
            # Create cost matrix via broadcasting with large int
            ts_a, ts_b = np.array(ts_a), np.array(ts_b)
            M, N = len(ts_a), len(ts_b)
            cost = sys.maxsize * np.ones((M, N))

            # Initialize the first row and column
            cost[0, 0] = d(ts_a[0], ts_b[0])
            for i in range(1, M):
                cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

            for j in range(1, N):
                cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

            # Populate rest of cost matrix within window
            for i in range(1, M):
                for j in range(max(1, i - self.max_warping_window), min(N, i + self.max_warping_window)):
                    choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                    cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

            return cost[-1, -1]
        else:
            return fastdtw(np.array(ts_a), np.array(ts_b), dist=d)[0]

    def _dist_matrix(self, x, y, logger):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]

        y : array of shape [n_samples, n_timepoints]

        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """

        # Compute the distance matrix
        dm_count = 0

        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if np.array_equal(x, y):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)

            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, :: self.subsample_step], y[j, :: self.subsample_step])

                    dm_count += 1
            # Convert to squareform
            dm = squareform(dm)
            return dm

        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)[0]
            y_s = np.shape(y)[0]
            dm = np.zeros((x_s, y_s))

            for i in range(0, x_s):
                logger.info(f"Progress - {i}/{x_s}")
                for j in range(0, y_s):
                    dm[i, j] = self._dtw_distance(x[i, :: self.subsample_step], y[j, :: self.subsample_step])
            return dm

    def predict(self, x, logger=logging):
        """Predict the class confidence scores for each k sample on
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified

        Returns
        -------
          1 arrays representing the confidence score for each n neighbor
        """

        dm = self._dist_matrix(x, self.x, logger)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, : self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]

        # Model Label
        num_classes = len(self.classes)
        results = np.apply_along_axis(np.bincount, axis=1, arr=knn_labels, minlength=num_classes)
        return results / self.n_neighbors
