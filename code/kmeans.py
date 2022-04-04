import numpy as np
from pyparsing import cpp_style_comment


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = [] # Initialized in initialize_centroids()
        self.dist = None
        self.sil = 0
        self.conv = 0

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        while iteration < self.max_iter:
            clustering = []
            self.dist = self.euclidean_distance(X,self.centroids)
            for i in range(len(X)):
                clustering.append(np.argmin(self.dist[i]))
            old_centroids = self.centroids.copy()
            self.update_centroids(clustering,X)
            new_centroids = self.centroids.copy() 
            optimised = True
            for o,p in zip(old_centroids,new_centroids):
                if(not np.array_equal(o,p)):
                    optimised = False  
                    break
            if(optimised):
                break
            iteration = iteration + 1
        self.sil = self.silhouette(clustering,X)
        self.conv = iteration
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        temp = {}
        for c in np.unique(clustering):
            temp[c] = []
        for d,c in zip(X,clustering):
            temp[c].append(d)
        for c in temp: 
            self.centroids[c] = np.average(temp[c], axis = 0)
            
    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            row_i = np.random.choice(X.shape[0],self.n_clusters,replace= False)
            self.centroids = X[row_i, :].copy()
        elif self.init == 'kmeans++':
            row_i = np.random.choice(X.shape[0],1,replace= False)
            self.centroids = X[row_i, :].copy()
            for k in range(1,self.n_clusters):
                #Calculate Distance between each data point and each centroid
                distances = self.euclidean_distance(X,self.centroids)
                weights = []
                #Find the closest not chosen centroid for each data point
                for i in range(len(X)):
                    weights.append(np.min(distances[i]))
                sum = np.sum(weights)
                weights = weights/sum
                next_centroid_index = np.random.choice(X.shape[0],1,replace= False, p=weights)
                next_centroid = X[next_centroid_index,:].copy()
                self.centroids = np.append(self.centroids,next_centroid,axis = 0)   

        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        dist = np.zeros([X1.shape[0],X2.shape[0]])
        for i in range(len(X1)):
            for j in range(len(X2)):
                dist[i][j] = np.linalg.norm(X1[i] - X2[j])
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        sil = 0
        for i in range(len(X)):
            sorted = np.sort(self.dist[i])
            a = sorted[0]
            b = sorted [1]
            s = (b-a)/max(a,b)
            sil = sil + s
        return sil/(len(X))




    