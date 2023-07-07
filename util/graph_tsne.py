from sklearn.manifold import TSNE

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import matplotlib.pyplot as plt
import numpy as np

import os


def graph_tsne(
    marker_gene,
    coordinate,
    cluster_id,
    target_cluster_id,
    expression_data,
    gene_list,
    random_state=0,
    tmp_dir=".",
):
    """
    Plot t-SNE visualization for a marker gene in a specific cluster.

    Parameters:
        marker_gene (str): The marker gene to visualize.
        coordinate (ndarray): The 2D spatial coordinates.
        cluster_id (ndarray): The cluster ID for each cell.
        target_cluster_id (int): The ID of the target cluster.
        expression_data (ndarray): The gene expression data.
        gene_list (ndarray): The list of gene symbols.
        random_state (int, optional): Random state for t-SNE. Default is 0.

    Returns:
        None. The plot is displayed and saved as a PNG file.
    """
    # Create a t-SNE model
    model = TSNE(n_components=2, random_state=random_state)

    # Flatten cluster_id
    cluster_id = np.ravel(cluster_id)

    # Calculate t-SNE coordinates for all cells
    tsne_coordinates = model.fit_transform(coordinate)

    # Set up the plot
    _, ax = plt.subplots()

    # Plot target cluster in grey
    target_cluster = tsne_coordinates[cluster_id == target_cluster_id]
    ax.scatter(
        target_cluster[:, 0], target_cluster[:, 1], color="grey", s=1, alpha=0.25
    )

    # Find cells in the target cluster that express the marker gene
    marker_gene_index = np.where(gene_list == marker_gene)[
        0
    ]  # find index of marker gene
    marker_gene_expression = (
        expression_data[:, marker_gene_index].flatten() > 0
    )  # find cells that express the marker gene

    print(cluster_id.shape)
    print(marker_gene_expression.shape)

    marker_gene_in_target_cluster = np.logical_and(
        cluster_id == target_cluster_id, marker_gene_expression
    )

    # Plot these cells in red
    marker_gene_coordinates = tsne_coordinates[marker_gene_in_target_cluster.ravel(), :]
    ax.scatter(
        marker_gene_coordinates[:, 0],
        marker_gene_coordinates[:, 1],
        color="red",
        s=1,
    )

    # Labels and title
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE for marker gene {marker_gene} in cluster {target_cluster_id}")

    # Save figure in png format
    file_name = f"tsne_{marker_gene}_c-{target_cluster_id}.png"
    file_path = os.path.join(tmp_dir, file_name)
    plt.savefig(file_path)

    # Optionally, you can still show the plot
    # plt.show()
