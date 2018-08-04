import numpy as np
import time
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn import metrics
import scipy
from sklearn.linear_model import RidgeClassifier 
# coding: utf-8

# In[ ]:


# https://github.com/matejbalog/mondrian-kernel/blob/master/random_binning.py
class RandomBinning(object):
    def __init__(self, D, M):
        """ Sets up a random binning object for the isotropic Laplacian kernel in D dimensions.
         A random binning object is a 3-tuple (widths, shifts, keys) where
         - widths is a list of D reals, specifying bin widths in each input dimension
         - shifts is a list of D reals, specifying bin shifts
         - keys is a dictionary int -> int giving sequential numbers to non-empty bins
        """

        self.widths = [np.array([np.random.gamma(shape=2, scale=1.0) for _ in range(D)]) for _ in range (M)]
        self.shifts = [np.array([np.random.uniform(low=0.0, high=width) for width in widths]) for widths in self.widths]
        self.keys = {}
        self.C = 0
        self.M = M
        self.D = D

    def get_features(self, X, M=None, expand=True):
        """ Returns unnormalized Random binning features for the provided datapoints X (one datapoint in each row).
        :param X: Matrix of dimensions NxD, containing N datapoints (one in each row).
        :param expand: Specifies whether new features should be created if a datapoint lies in a bin
         that has been empty so far. (True for training, False for testing.)
        :return: Sparse binary matrix of dimensions NxC, where C is the number of generated features.
        Each row is the feature expansion of one datapoint and contains at most M ones.
        """
        N = np.shape(X)[0]

        if M is None:
            M = self.M
        assert M <= self.M

        # stacking experiment
        X_stack = np.tile(X, self.M)
        shifts_stack = np.concatenate(self.shifts)
        widths_stack = np.concatenate(self.widths)
        X_coordinates = np.ceil((X_stack - shifts_stack) / widths_stack).astype(int)

        # compute indices
        row_indices = []
        col_indices = []
        X_coordinates.flags.writeable = False
        feature_from_repetition = []
        for m in range(M):
            X_coords = X_coordinates[:, (self.D*m):(self.D*(m+1))]
            X_coords.flags.writeable = False
            for n, coordinates in enumerate(X_coords):
                coordinates.flags.writeable = False
                #h = hash(coordinates.data)
                h = tuple(coordinates.tolist())
                if (m, h) in self.keys:
                    row_indices.append(n)
                    col_indices.append(self.keys[(m, h)])
                elif expand:
                    row_indices.append(n)
                    col_indices.append(self.C)
                    self.keys[(m, h)] = self.C
                    feature_from_repetition.append(m)
                    self.C += 1

        # construct features
        values = [1]*len(row_indices)
        Z = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=(N, self.C))
        return Z.tocsr(), np.array(feature_from_repetition)


    def random_binning_features(X, R_max):
        D = X.shape[1]
        rb = RandomBinning(D, R_max)
        return rb.get_features(X)


    def evaluate_random_binning(X, y, X_test, y_test, M, task):
        # construct random binning features
        start_time = time.time()
        rb = RandomBinning(X.shape[1], M)
        Z, _ = rb.get_features(X) / np.sqrt(M)
        Z_test, _ = rb.get_features(X_test, expand=False) / np.sqrt(M)
        if(task=='classification'):
            clf = RidgeClassifier(alpha=0.0001, solver='lsqr')
            clf.fit(Z, y) 
            y_pred = clf.predict(Z_test)
            error_test = (0.5 - np.dot(np.sign(y_test), y_pred)/len(y_test)/2)*100
            print("--- %s seconds ---" % (time.time() - start_time))
            print("C = %d; error_test = %.2f" % (np.shape(Z)[1], error_test)+'%')
        elif(task=='regression'):
            clf = Ridge(alpha=0.01, solver='lsqr',random_state=42)
            clf.fit(Z, y) 
            y_pred = clf.predict(Z_test)
            error_test = np.linalg.norm((y_test-y_pred))/np.linalg.norm(y_test)*100
            print("--- %s seconds ---" % (time.time() - start_time))
            print("C = %d; error_test = %.2f" % (np.shape(Z)[1], error_test)+'%')
        else:
            error_test = 'error!'
            print('No such a task, please check the task name!')
        return error_test

