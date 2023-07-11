# t-Distributed Stochastic Neighbor Embedding (t-SNE)

t-SNE is a machine learning algorithm used for dimensionality reduction that is particularly well suited for the visualization of high-dimensional datasets.

## Basic Concept

The goal of t-SNE is to take a set of points in a high-dimensional space and find a faithful representation of those points in a lower-dimensional space, typically a 2D or 3D space for visualization.

t-SNE works by first calculating the probability of similarity of points in high-dimensional space and then calculating the similarity of those points in the corresponding low-dimensional space. The algorithm iteratively minimizes the difference between these two probability distributions with respect to the locations of the points in the low-dimensional space.

## Algorithm Steps

1. **Compute pairwise affinities** with Gaussian distribution in high dimensional space. This creates a probability distribution where similar objects have a higher probability of being picked than dissimilar ones.

2. **Compute pairwise affinities in low dimensional map**. In the low-dimensional representation, t-SNE uses a Student-t distribution (hence the "t" in t-SNE) to compute the pairwise affinities. The Student-t distribution has heavier tails than the Gaussian distribution, which helps to prevent the "crowding problem" where points in the low-dimensional representation are closer together than their counterparts in the high-dimensional space.

3. **Minimize divergence**. The Kullback-Leibler (KL) divergence is used as a cost function to measure the mismatch between the two probability distributions. The positions of the points in the lower-dimensional space are adjusted in a way that minimizes the KL divergence.

## Key Advantages

- t-SNE is excellent at visualizing clusters or groups of data points and retaining the structure of the data.
- It's also good at revealing the structure at many different scales on a single map.

## Key Limitations

- The interpretation of the distance between clusters or the relative sizes of clusters in a t-SNE plot can be misleading.
- It has a computational complexity of O(n^2), making it slow for large datasets.
- The results are not stable, different runs with the same hyperparameters can produce different results due to the random initialization.

---

## Hyperparameters

t-SNE has several important hyperparameters that need to be tuned:

- **Perplexity**: This parameter can be interpreted as a knob that sets the number of effective nearest neighbors. It balances attention between local and global aspects of the data, and is usually chosen between 5 and 50.

- **Learning rate**: This parameter scales the gradients during optimization. If it's too low, the data points hardly move and the algorithm takes a long time. If it's too high, the data points move too much and you don't get a useful result.

- **Number of iterations**: More iterations allow the algorithm to find better embeddings, but increase the computational cost.

Remember, like all machine learning algorithms, t-SNE isn't perfect and has its own set of assumptions and limitations, so use with caution and validate its results with other methods when appropriate.
