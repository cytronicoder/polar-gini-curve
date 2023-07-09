"""
Functions for computing and plotting Gini coefficients and Lorenz curves, as well as the t-SNE graph.
"""
import os
import numpy as np
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Use a non-interactive backend


def compute_gini(population, value, plot=False):
    """Compute the Gini coefficient and plot the Lorenz curve."""
    assert len(population) == len(value), "Expected vectors of equal length."

    population = np.append(0, population)  # Pre-append a zero
    value = np.append(0, value)  # Pre-append a zero

    # Filter out NaNs
    valid_data = ~np.isnan(population) & ~np.isnan(value)
    if np.sum(valid_data) < 2:
        print("Warning: Not enough data")
        return np.nan, np.nan, np.nan

    population = population[valid_data]
    value = value[valid_data]

    assert np.all(population >= 0) and np.all(
        value >= 0
    ), "Expected nonnegative vectors."

    # Process input
    weighted_values = value * population
    sorted_indices = np.argsort(value)
    population = population[sorted_indices]
    weighted_values = weighted_values[sorted_indices]
    population = np.cumsum(population)
    weighted_values = np.cumsum(weighted_values)
    relative_population = population / population[-1]
    relative_values = weighted_values / weighted_values[-1]

    # Compute Gini coefficient
    gini = 1 - np.sum(
        (relative_values[:-1] + relative_values[1:]) * np.diff(relative_population)
    )

    # Compute Lorenz curve
    lorenz_curve = np.column_stack([relative_population, relative_values])
    curve_points = np.column_stack([population, weighted_values])

    if plot:  # ... plot it?
        plt.fill_between(
            relative_population, relative_values, color=[0.5, 0.5, 1.0]
        )  # Lorenz curve
        plt.plot([0, 1], [0, 1], "--k")  # 45 degree line
        plt.axis("tight")
        plt.axis("equal")  # Equally long axes
        plt.grid()
        plt.title(f"Gini coefficient = {gini}")
        plt.xlabel("Share of population")
        plt.ylabel("Share of value")
        plt.show()

    return gini, lorenz_curve, curve_points


def plot_tsne(
    marker_gene,
    coordinates,
    cluster_ids,
    target_cluster_id,
    expression_data,
    gene_list,
    output_dir=".",
    random_state=0,
):
    """Plot t-SNE visualization for a marker gene in a specific cluster."""
    tsne_model = TSNE(n_components=2, random_state=random_state)

    # Flatten cluster_ids
    cluster_ids = np.ravel(cluster_ids)

    # Calculate t-SNE coordinates for all cells
    tsne_coordinates = tsne_model.fit_transform(coordinates)

    # Set up the plot
    _, ax = plt.subplots()

    # Plot target cluster in grey
    target_cluster = tsne_coordinates[cluster_ids == target_cluster_id]
    ax.scatter(
        target_cluster[:, 0], target_cluster[:, 1], color="grey", s=1, alpha=0.25
    )

    # Find cells in the target cluster that express the marker gene
    gene_index = np.where(gene_list == marker_gene)[0]  # Index of marker gene
    gene_expression = (
        expression_data[:, gene_index].flatten() > 0
    )  # Cells that express the gene

    cells_in_cluster_with_gene = np.logical_and(
        cluster_ids == target_cluster_id, gene_expression
    )

    # Plot these cells in red
    cells_coordinates = tsne_coordinates[cells_in_cluster_with_gene.ravel(), :]
    ax.scatter(cells_coordinates[:, 0], cells_coordinates[:, 1], color="red", s=1)

    # Labels and title
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE for marker gene {marker_gene} in cluster {target_cluster_id}")

    # Save figure in PNG format
    file_name = f"tsne_{marker_gene}_c-{target_cluster_id}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)


def plot_gini(
    marker_gene,
    coordinates,
    cluster_ids,
    target_cluster_id,
    expression_data,
    gene_list,
    output_dir=".",
):
    """Plot Gini coefficients for each cluster."""
    cluster_ids = cluster_ids.flatten()  # Flatten the cluster_ids array

    cluster_indices = np.where(cluster_ids == target_cluster_id)[
        0
    ]  # Get all cells in target_cluster_id
    cluster_coordinates = coordinates[
        cluster_indices, :
    ]  # Get the spatial coordinates of cells in target_cluster_id
    cluster_labels = np.ones(len(cluster_indices))  # Label these cells as '1'

    marker_gene_index = np.where(gene_list == marker_gene)[0]
    marker_gene_expression = expression_data[:, marker_gene_index]
    cells_indices = np.where(
        (cluster_ids == target_cluster_id) & (marker_gene_expression > 0)
    )[
        0
    ]  # Find cells expressing marker_gene in target_cluster_id

    cells_coordinates = coordinates[
        cells_indices, :
    ]  # Get the spatial coordinates of cells expressing marker_gene in target_cluster_id
    cells_labels = 2 * np.ones(len(cells_indices))  # Label these cells as '2'

    data = np.vstack((cells_coordinates, cluster_coordinates))
    labels = np.hstack((cells_labels, cluster_labels))

    num_cluster = len(np.unique(labels))

    # Get the angle list with resolution of 1000
    resolution = 1000
    angle_list = np.pi * np.linspace(0, 360, resolution) / 180

    colors = ["red", "gray"]
    alphas = [1, 0.25]

    # For each cluster, get the list of gini corresponding to each angle
    for idx, cluster in enumerate(range(1, num_cluster + 1)):
        coords = data[labels == cluster]
        gini_values = np.zeros(len(angle_list))

        for i, angle in enumerate(angle_list):
            val = np.dot(coords, [np.cos(angle), np.sin(angle)])
            val -= np.min(val)
            gini_values[i] = compute_gini(np.ones(len(val)), val)[0]

        plt.plot(
            angle_list * 180 / np.pi, gini_values, color=colors[idx], alpha=alphas[idx]
        )

    plt.xlabel("Angle (degree)")
    plt.ylabel("Gini coefficient")
    plt.title(
        f"Gini coefficient plot for marker gene {marker_gene} in cluster {target_cluster_id}"
    )
    plt.legend(["Cells with gene", "All cells"], loc="lower right")
    plt.grid()

    # Save figure in PNG format
    file_name = f"gini_{marker_gene}_c-{target_cluster_id}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()