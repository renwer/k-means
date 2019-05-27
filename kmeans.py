import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('iris.csv')
clusters_count = int(input("Enter the desired number of clusters: \n"))

#initializing the array for holding centroids
centroids = np.random.rand(clusters_count, df.shape[1] - 1)

#initialize the array of indicators whether the centroids were relocated
centroids_relocated = [False] * len(centroids)

labels = np.random.randint(clusters_count+1, size=df.shape[0])

df['cluster'] = labels
print(labels)

for i in range(len(centroids)):
    cluster = df.loc[df['cluster'] == i]
    cluster_data = cluster.values[:, :4]
    if cluster_data.shape[0] == 0:
        centroids[i] = df.values[:, :4].mean(0) + np.random.rand(df.shape[1] - 2)
    else:
        centroids[i] = cluster_data.mean(0)

def plot_current_state():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df.values[:, 0], df.values[:, 1], df.values[:, 2], c='blue')
    for centroid in centroids:
        ax.scatter(centroid[0], centroid[1], centroid[2], c='r', marker='x')
    fig.show()


def print_cluster_purity():
    clusters = df.groupby(['cluster'])
    stats_string = ''
    for key, cluster in clusters:
        cluster_number = cluster.iloc[0].loc['cluster'] #the same everywhere, so why bother
        stats_string += '\ncluster {}: '.format(cluster_number)
        species = cluster.groupby(['Species'])
        for sp_key, specie in species:
            specie_name = specie.iloc[0].loc['Species'] #same as on 33
            specie_presence = (specie.size / cluster.size) * 100
            stats_string += '{}: {}%; '.format(specie_name, specie_presence)
    print(stats_string)


while True:

    centroid_relocations = [0]*clusters_count
    #assign all datapoints to clusters
    for idx, row in df.iterrows():
        distances = []
        for i in range(len(centroids)):
            mod_row = np.array(row[:4])
            #append the centroid relocation distance per each cluster
            distances.append(np.linalg.norm(centroids[i] - mod_row))
        labels[idx] = distances.index(min(distances))

    df['cluster'] = labels
    plot_current_state()

    #Find the cluster means, reassign cluster centroids
    for i in range(len(centroids)):
        cluster = df.loc[df['cluster'] == i]

        if cluster.shape[0] == 0:
            print('empty cluster warning')

        cluster_data = cluster.values[:, :4]
        if cluster_data.shape[0] != 0:
            cluster_mean = cluster_data.mean(0)
        else:
            flip = 1
            if np.random.random() > 0.5:
                flip = -1
            cluster_mean = df.values[:, :4].mean(0) + np.random.rand(df.shape[1] - 2) * flip * 1.5

        if cluster_data.shape[0] != 0:
            centroid_relocations[i] += np.linalg.norm(centroids[i] - cluster_mean)
            centroids_relocated[i] = True
        else:
            centroid_relocations[i] += 0

        centroids[i] = cluster_mean

        wcss = 0
        for k in range(len(centroids)):
            cluster = df.loc[df['cluster'] == k]
            cluster_data = cluster.values[:, :4]
            for vector in cluster_data:
                wcss += np.linalg.norm(centroids[k] - vector) ** 2
    print('Within-cluster sum of squares for all clusters is {}'.format(wcss))
    print_cluster_purity()

    print()
    if sum(centroid_relocations) == 0 and (False not in centroids_relocated):
        break

