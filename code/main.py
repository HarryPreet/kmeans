import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=4)
    parser.add_argument('--data', type=str, default='../data/data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    print("Reading Data")
    heart = read_data(data_path)
    print("Preprocessing Data")
    heart = preprocess_data(heart)
    print("Doing PCA")
    X = PCA(heart.X, 100)
    km = KMeans(n_clusters= n_classifiers, init='random')
    print("Performing K means clustering")
    labels = km.fit(X)
    X = PCA(heart.X, 2)
    PCA_components = pd.DataFrame(X,columns = ['PC1', 'PC2'])
    PCA_components['labels'] = labels
    visualize_cluster(x=0,y=0,clustering=PCA_components)
    

def visualize_cluster(x, y, clustering):
    print('Visualising')
    label0 = clustering[clustering['labels'] == 0]
    label1 = clustering[clustering['labels'] == 1]
    label2 = clustering[clustering['labels'] == 2]
    label3 = clustering[clustering['labels'] == 3]
    fig = plt.figure()
    plt.title('K-means Random Intialisation with k = 5\n',
                fontsize = 14, fontweight ='bold')
    plt.scatter(label0['PC1'] , label0['PC2'] , color = 'red')
    plt.scatter(label1['PC1'] , label1['PC2'] , color = 'green')
    plt.scatter(label2['PC1'] , label2['PC2'] , color = 'blue')
    plt.scatter(label3['PC1'] , label3['PC2'] , color = 'yellow')
    plt.show()


if __name__ == '__main__':
    main()
