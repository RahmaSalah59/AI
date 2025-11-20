# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 05:29:20 2025

@author: ascom
"""


import pandas as pd
dataset = pd.read_csv("Mall_Customers.csv")

x = dataset.iloc[:,[3,4]].values
#x = pd.DataFrame(x)

from sklearn.cluster import KMeans

wcss =[]

for i in range (1,21):
    kmeans = KMeans(n_clusters = i ,init = "k-means++", random_state = 42 )
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt

plt.plot(range(1,21),wcss)
plt.title('the Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters = 5 ,init = "k-means++", random_state = 42 )
y_kmeans = kmeans.fit_predict(x)
x[y_kmeans == 0]

# 52 Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

cent = kmeans.cluster_centers_