from random import random, randint
from xmlrpc.client import boolean
import numpy as np

import matplotlib.pyplot as plt

class kmean:
    def __init__(self, M: np.array, k: int, random_centroids : bool = False):
        '''
        K-Means Clustering
        M - Numpy Array of n x m size (no labels)
        k - Int - Number of sets to split dataset into
        random_centroids - bool - whether intial centroids should be random or not
        '''

        # Dimension of the data
        self.x_n = M.shape[1]
        # Input data
        self.data = M
        # Predicted number of features 
        self.k = k

        # random_centroids
        self.random_centroids = random_centroids

        # Init mean
        self.m = None
        # Init sets
        self.sets = None

    def _assign_to_sets(self, m: np.array) -> list:
        sets = []
        distances = np.zeros([self.data.shape[0], m.shape[0]+1])
        counter = 0

        for m_i in m:
            distances[:, counter] = np.sum(np.square(self.data - m_i), axis = 1)
            counter += 1
            
        set_indexes = np.argmax(distances, axis = 1) + 1
        set_indexes = set_indexes.reshape(set_indexes.shape[0], 1)
        data_arr_w_set = np.append(self.data, set_indexes, axis = 1)

        for ki in range(1, self.k + 1):
            sets.append(data_arr_w_set[data_arr_w_set[:, -1] == ki][:, :-1])
        return sets

    def _update_means(self, m: np.array, sets: list) -> np.array:
        i = 0
        for subset in sets:
            means = np.dot(subset.T, np.ones([subset.T.shape[1], 1]))/subset.shape[0]
            m[i] = means.reshape(-1).tolist()
            i += 1
        return m

    def _terminal_condition(self, u: np.array, old_u: np.array) -> boolean:
        return sorted(np.round(u, 3).tolist()) == sorted(np.round(old_u, 3).tolist())

    def _alter_sets_for_return(self, list_of_array: list) -> np.array:
        counter = 1
        final_M = []
        for arri in list_of_array:
            final_M.append(np.append(arri, counter*np.ones((arri.shape[0], 1)), axis = 1))
            counter += 1
        return np.concatenate(final_M)
        
    def _init_centroids(self):
        if self.random_centroids:
            return np.array([np.array([random()*np.mean(self.data[:, i]) for i in range(self.x_n)]) for j in range(self.k)])
        else: 
            rand_index = []
            while len(rand_index) != self.k:
                rand_val = randint(0, self.data.shape[0])
                if rand_val not in rand_index:
                    rand_index.append(rand_val)
            return self.data[rand_index, :]

    def fit(self):
        print("== Starting ==")
        print("== Iteration 1 ==")
        # 1. Init Centroid 
        m = self._init_centroids()
    
        # 2. Assign data to sets
        sets = self._assign_to_sets(m)

        # 3. Update Mean
        old_m = m.copy()
        m = self._update_means(m, sets)

        it = 2

        while not self._terminal_condition(m, old_m):
            print(f"== Iteration {str(it)} ==")
            it += 1
            # 2. Assign data to sets
            sets = self._assign_to_sets(m)

            # 3. Update Mean
            old_m = m.copy()
            m = self._update_means(m, sets)

        self.m = m 
        self.sets = sets
        return self.m, self._alter_sets_for_return(self.sets)


class kPlotter:
    def __init__(self, data: np.array, centroids: np.array, title: str = None):
        '''
        Plot Kmeans Data
            data - (np.array) n x 2 dimension data to plot with last column being set
            centroids - (np.array) no of centroids x 2
            title - (str) title 
        '''
    
        self.data = data
        self.centroids = centroids
        self.title = title

        self.color = None
        self._plot()

    def _add_scatter(self, seti):
        c = next(self.color)
        plt.scatter(self.data[self.data[:, -1] == seti][:, 0], self.data[self.data[:, -1] == seti][:, 1], c=c)
        
    def _add_means(self):
        for irow in range(self.centroids.shape[0]):
            c = next(self.color)
            plt.scatter(self.centroids[irow, :][0], self.centroids[irow, :][1], c=c, marker="*")


    def _plot(self):
        unique_sets = np.unique(self.data[:, -1]).tolist()
        self.color = iter(plt.cm.rainbow(np.linspace(0, 1, len(unique_sets) + self.centroids.shape[1])))

        for seti in unique_sets:
            self._add_scatter(seti)
        
        self._add_means()

        if self.title is not None:
            plt.title(self.title)

        plt.show()