

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
""" Partie1 :Creatiion de base de données Artificielle"""
from sklearn.datasets.samples_generator import make_blobs
X,y=make_blobs(n_samples=600,n_features=2,centers=4)
plt.scatter(X[:,0],X[:,1])

import matplotlib.pyplot as plt

""" Partie 2: Eblow method  """
Sum_of_squared_distances = []
K = range(1,8)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

"""Partie 3: cluster the data into four clusters using Kmeans"""
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_pred = kmeans.predict(X)
""" plot the cluster assignments and cluster centers"""
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:, 0],   
            kmeans.cluster_centers_[:, 1],
            marker='^', 
            c=[0, 1, 2, 3], 
            s=100, 
            linewidth=2,
            cmap="plasma")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()