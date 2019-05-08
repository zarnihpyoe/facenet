import numpy as np
import matplotlib.pyplot as plt

def PCA(X, Y):
    xbar = X.mean(axis=1).reshape(-1,1)
    Xtilde = X - xbar
    _, v = np.linalg.eigh(Xtilde.dot(Xtilde.T))
    d1 = v[:,-1]        # vector along which data varies the most
    d2 = v[:,-2]        # vector along which data varies second most
    plt.scatter(X.T.dot(d1), X.T.dot(d2), marker='o', c=Y, s=4)
    # plt.scatter(X[:, 398:].T.dot(d1), X[:, 398:].T.dot(d2), marker='.', c=Y[398:], s=2)
    plt.show()

# embeddings, labels = np.load('/Users/zarnihpyoe/Projects/instag/faces/embeddings.npy')
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/embeddings.npy')
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.2/embeddings.npy')
# embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data/0.3/embeddings.npy')
embeddings, labels = np.load('/Users/zarnihpyoe/wpi/mqp/data5/embeddings.npy')

# embeddings = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/ext_embeddings.npy')
# labels = np.load('/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj/labels.npy')


print(embeddings.shape, len(labels))
PCA(embeddings, labels)
