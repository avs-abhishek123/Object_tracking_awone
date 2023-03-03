# Meanshift

Meanshift is a clustering algorithm that assigns the datapoints to the clusters iteratively by shifting points towards the mode. The mode can be understood as the highest density of datapoints (in the region, in the context of the Meanshift). As such, it is also known as the mode-seeking algorithm.

Given a set of datapoints, the algorithm iteratively assign each datapoint towards the closest cluster centroid. The direction to the closest cluster centroid is determined by where most of the points nearby are at. So each iteration each data point will move closer to where the most points are at, which is or will lead to the cluster center. When the algorithm stops, each point is assigned to a cluster.

Unlike the popular K-Means algorithm, meanshift does not require specifying the number of clusters in advance. The number of clusters is determined by the algorithm with respect to the data.

The blue datapoints are the initial datapoints and red are the positions of those datapoints at each iteration. Description for each step:

Iteration:

1. Initial state. The red and blue datapoints overlap completely in the first iteration before the Meanshift algorithm starts.
2. End of iteration 1. All the red datapoints move closer to clusters. Looks like there will be 4 clusters.
3. End of iteration 2. The clusters of upper right and lower left seems to have reached convergence just using two iterations. The center and lower right clusters looks like they are merging, since the two centroids are very close.
4. End of iteration 3. No change in the upper right and lower left centroids. The other two centroids’ have pulled each other together as the datapoints affect each clusters. This is a signature of Meanshift, the number of clusters are not pre-determined.
5. End of iteration 4. All the clusters should have converged.
6. End of iteration 5. All the clusters indeed have no movement. The algorithm stops here since no change is detected for all red datapoints.

The algorithm’s runtime complexity is O(KN2), where N is the number of datapoints and K is the number of iterations of Meanshift.

### K-Means VS Meanshift
Meanshift looks very similar to K-Means, they both move the point closer to the cluster centroids. One may wonder: How is this different from K-Means? K-Means is faster in terms of runtime complexity!

The key difference is that Meanshift does not require the user to specify the number of clusters. In some cases, it is not straightforward to guess the right number of clusters to use. In K-Means, the output may end up having too few clusters or too many clusters to be useful. At the cost of larger time complexity, Meanshift determines the number of clusters suitable to the dataset provided.

Another commonly cited difference is that K-Means can only learn circle or ellipsoidal clusters. However, this is not true. The reason that Meanshift can learn arbitrary shapes is because the features are mapped to another higher dimensional feature space through the kernel. The arbitrary shapes are due to the algorithm finding circle or ellipsoidal clusters in higher dimensional feature space. When the features are mapped back to 1D/2D/3D, the resulting clusters look like strange shapes. This is also the trick as used in Support Vector Machines.

A traditional K-means does not use kernels, but Kernel K-means is available. Kernel K-Means is useful if 1) the number of clusters is known or can be reasonably estimated, and 2) dataset needs learning non-ellipsoidal cluster shapes. So, you can enjoy the better runtime complexity of K-Means and learn arbitrary clusters if you can determine the number of clusters to use.
