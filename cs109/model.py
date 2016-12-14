import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph


def density_last(X, n_neighbors, metric):
    if (X.shape[0] <= n_neighbors):
        return 1.0 / n_neighbors
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric, mode='distance', n_jobs=-1)
    # total distance to nearest neighbors
    t1 = graph[-1].sum()
    # total distance fromm the nearest neighbors to their nearest neighbors
    t2 = graph[graph[-1].indices].sum()
    # strange if t1 is big and t2 is small
    # meaning density arounfd this point is less then density around it's neighbors
    # possible to observe identical vectors
    return 1.0 / n_neighbors if t1 * t2 == 0 else t1 / t2


def proximity_last(X, n_neighbors, metric):
    if (X.shape[0] <= n_neighbors):
        return 1.0 / n_neighbors
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric, mode='distance', n_jobs=-1)
    # maximum distance in the the neighborhood
    return np.max(graph[-1])


def martingale(m, p, epsilon):
    return m * epsilon * np.power(p, epsilon - 1.0)


def p_value(A, a):
    return (float(np.sum(A > a)) + np.random.uniform() * float(np.sum(A == a))) / float(A.size)


# Online Anomaly Detection in time series
#      uses KNN-based relative density as the strangeness function
# Input: 
#      threshold - float, optional (default value 2.0)
#           the anomaly reported when the threshold is breached
#      epsilon - float, optional (default value 0.92)
#           Value from 0.0 t0 1.0, used in martingale calculation
#      n_neighbors - int, optional (default value 3)
#           The neighborhood size.
#      metric - string, optnonal (default value 'euclidean')
#           The distance metric used to calculate the k-Neighbors for each sample point
#      method - string , 'density' or 'proximity', default value 'density'
#           Method of calculating strangeness function. 
#           When 'density' the strangeness is calculated as density around the point 
#           devided by total density around the neighborhood
#           When 'proximity' the strangeness is calculated as maximum distance in the the neighborhood
#      anomaly - string , 'level' or 'change', default value 'level'
#           Method of using the threshold
#           When 'level' the martngale value is compared with the threshold
#           When 'change' the change in martngale value is compared with the threshold


class KNNAnomalyDetector():
    def __init__(self, threshold=2.0, epsilon=0.92, n_neighbors=3, metric='euclidean', method='density',
                 anomaly='level'):
        self.threshold = threshold
        self.epsilon = epsilon
        self.n_neighbors = n_neighbors
        self.metric = metric
        assert method == 'density' or method == 'proximity'
        self.method = method
        assert anomaly == 'level' or anomaly == 'change'
        self.anomaly = anomaly
        # default score (used when not possible to calcuate normal score)
        self.def_score = 1.0 / self.n_neighbors
        self.observations = []
        self.A = []
        self.M = 1.0

    def score_last(self, X):
        if self.method == 'density':
            return density_last(X, self.n_neighbors, self.metric)
        return proximity_last(X, self.n_neighbors, self.metric)

    def observe(self, x):
        self.observations.append(x)
        X = StandardScaler().fit_transform(np.array(self.observations))
        # strangeness measure
        a = self.score_last(X)
        self.A.append(a)
        A = np.array(self.A)
        # randomized p-value
        p = p_value(A, a)
        # randomized power martingale
        m = martingale(self.M, p, self.epsilon)
        if self.anomaly == 'level':
            is_anomaly = (m > self.threshold)
        else:
            is_anomaly = ((m - self.M) > self.threshold)
        if is_anomaly:
            self.M = 1.0
            self.observations = []
        else:
            self.M = m
        return [a, p, m, is_anomaly]
