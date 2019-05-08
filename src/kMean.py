from sklearn.cluster import KMeans
import numpy as np
from itertools import product, permutations
import matplotlib.pyplot as plt

# def test_accuracy(X, Y):
#     # n: number of clusters
#     n = len(list(set(Y)))
#     print('n =', n)
#     kmeans = KMeans(n_clusters=n).fit(X)
#     Y_hat = kmeans.predict(X)

#     corrects = []
#     accuracy_percents = []
    
#     for i,j in product(range(n), repeat=2):
#         num_correct = 0
#         for y,y_hat in zip(Y, Y_hat):
#             if y == y_hat or (y == i and y_hat == j) or (y == j and y_hat == i):
#                 num_correct += 1
        
#         corrects.append(num_correct)
#         accuracy_percents.append(100 * num_correct/len(Y))
#     # print(corrects)
#     # print(accuracy_percents)
#     print(max(corrects))
#     print(max(accuracy_percents))

def test_accuracy(X, Y):
    # n: number of clusters
    n = len(list(set(Y)))
    kmeans = KMeans(n_clusters=n).fit(X)
    Y_hat = kmeans.predict(X)
    
    # NOTE: centroids.shape = (2,512)
    centroids = kmeans.cluster_centers_

    max_correct = 0
    max_fixed_Y_hat = []
    max_fixed_centroids = 0
    
    for x in permutations(range(n)):
        num_correct = 0
        fixed_Y_hat = [x[y_hat] for y_hat in Y_hat]

        for y,y_hat in zip(Y, fixed_Y_hat):
            if y == y_hat:
                num_correct += 1
        
        if num_correct > max_correct:
            max_correct = num_correct
            max_fixed_Y_hat = fixed_Y_hat
            max_fixed_centroids = centroids[list(x)]
    
    return max_correct, max_fixed_Y_hat, max_fixed_centroids
    

def multi_test_accuracy(X, Y):
    max_correct = 0
    max_Y_hat = []
    max_centroids = 0

    for _ in range(10):
        correct, Y_hat, centroids = test_accuracy(X, Y)
        if correct > max_correct:
            max_correct = correct
            max_Y_hat = Y_hat
            max_centroids = centroids
    
    print('{} correct predictions out of {}'.format(max_correct, len(Y)))
    print('{} accuracy'.format(max_correct/len(Y)))
    # print(100 * max_correct/len(Y), max_correct)

    Y_diff = np.array([y-y_hat == 0 for y, y_hat in zip(Y, max_Y_hat)])
    PCA(X.T, Y_diff)

    # NOTE: centroids_embeddings saved as (512,n)
    np.save('/Users/zarnihpyoe/wpi/mqp/data5/centroids_embeddings.npy', max_centroids.T)


def PCA(X, Y):
    xbar = X.mean(axis=1).reshape(-1,1)
    Xtilde = X - xbar
    _, v = np.linalg.eigh(Xtilde.dot(Xtilde.T))
    d1 = v[:,-1]        # vector along which data varies the most
    d2 = v[:,-2]        # vector along which data varies second most
    plt.scatter(X.T.dot(d1), X.T.dot(d2), marker='o', c=Y, s=4)
    plt.show()

# X, Y = np.load('/Users/zarnihpyoe/Projects/instag/faces/embeddings.npy')
# X, Y = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/embeddings.npy')
# X, Y = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.2/embeddings.npy')
# X, Y = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3/embeddings.npy')
X, Y = np.load('/Users/zarnihpyoe/wpi/mqp/data5/embeddings.npy')

# X = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/ext_embeddings.npy')
# Y = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/labels.npy')

X = X.T
# test_accuracy(X, Y)
multi_test_accuracy(X, Y)
