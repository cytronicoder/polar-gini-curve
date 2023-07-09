from sklearn.manifold import TSNE

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import os


def compute_gini(pop, val, makeplot=False):
    """
    Compute the Gini coefficient and plot the Lorenz curve.

    Parameters:
    pop (array-like): Population vector.
    val (array-like): Value vector.
    makeplot (bool, optional): Whether to plot the Lorenz curve. Default is False.

    Returns:
    gini (float): Gini coefficient.
    lorenz (ndarray): Lorenz curve.
    curve_points (ndarray): Coordinate points of the Lorenz curve.
    """

    assert len(pop) == len(val), "compute_gini expects two equally long vectors."

    pop = np.append(0, pop)  # pre-append a zero
    val = np.append(0, val)  # pre-append a zero

    isok = ~np.isnan(pop) & ~np.isnan(val)  # filter out NaNs
    if np.sum(isok) < 2:
        print("Warning: Not enough data")
        return np.nan, np.nan, np.nan

    pop = pop[isok]
    val = val[isok]

    assert np.all(pop >= 0) and np.all(
        val >= 0
    ), "compute_gini expects nonnegative vectors."

    # process input
    weighted = val * pop
    sorted_indices = np.argsort(val)
    pop = pop[sorted_indices]
    weighted = weighted[sorted_indices]
    pop = np.cumsum(pop)
    weighted = np.cumsum(weighted)
    relpop = pop / pop[-1]
    relz = weighted / weighted[-1]

    # Gini coefficient
    gini = 1 - np.sum((relz[:-1] + relz[1:]) * np.diff(relpop))

    # Lorentz curve
    lorenz = np.column_stack([relpop, relz])
    curve_points = np.column_stack([pop, weighted])

    if makeplot:  # ... plot it?
        plt.fill_between(relpop, relz, color=[0.5, 0.5, 1.0])  # the Lorentz curve
        plt.plot([0, 1], [0, 1], "--k")  # 45 degree line
        plt.axis(
            "tight"
        )  # ranges of abscissa and ordinate are by definition exactly [0,1]
        plt.axis("equal")  # both axes should be equally long
        plt.grid()
        plt.title(f"Gini coefficient = {gini}")
        plt.xlabel("Share of population")
        plt.ylabel("Share of value")
        plt.show()

    return gini, lorenz, curve_points


def graph_tsne(
    marker_gene,
    coordinate,
    cluster_id,
    target_cluster_id,
    expression_data,
    gene_list,
    tmp_dir=".",
    random_state=0,
):
    """
    Plot t-SNE visualization for a marker_gene gene in a specific cluster.

    Parameters:
        marker_gene (str): The marker_gene gene to visualize.
        coordinate (ndarray): The 2D spatial coordinates.
        cluster_id (ndarray): The cluster ID for each cell.
        target_cluster_id (int): The ID of the target cluster.
        expression_data (ndarray): The gene expression_data data.
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
    target_cluster_id = tsne_coordinates[cluster_id == target_cluster_id]
    ax.scatter(
        target_cluster_id[:, 0], target_cluster_id[:, 1], color="grey", s=1, alpha=0.25
    )

    # Find cells in the target cluster that express the marker_gene gene
    marker_gene_index = np.where(gene_list == marker_gene)[
        0
    ]  # find index of marker_gene gene
    marker_gene_expression = (
        expression_data[:, marker_gene_index].flatten() > 0
    )  # find cells that express the marker_gene gene

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
    ax.set_title(f"t-SNE for marker_gene gene {marker_gene} in cluster {target_cluster_id}")

    # Save figure in png format
    file_name = f"tsne_{marker_gene}_c-{target_cluster_id}.png"
    file_path = os.path.join(tmp_dir, file_name)
    plt.savefig(file_path)

    # Optionally, you can still show the plot
    # plt.show()


def graph_gini(
    marker_gene,
    coordinate,
    cluster_id,
    target_cluster_id,
    expression_data,
    gene_list,
    tmp_dir=".",
):
    """
    Plot Gini coefficients for each cluster.

    Parameters:
    coordinate (numpy array): The 2D spatial coordinates.
    cluster_id (numpy array): The ID for each cluster.
    expression_data (numpy array): The gene expression data.
    gene_list (numpy array): The list of gene symbols.
    marker_gene (str, optional): The marker gene to consider. Default is 'Actc1'.
    target_cluster_id (int, optional): The cluster number to plot. Default is 1.

    Returns:
    None. The plot is displayed.
    """

    cluster_id = cluster_id.flatten()  # Flatten the cluster_id array

    cluster_index = np.where(cluster_id == target_cluster_id)[
        0
    ]  # get all cells in target_cluster_id
    cluster_coor = coordinate[
        cluster_index, :
    ]  # get the spatial coordinate of cells in target_cluster_id
    cluster_lbl = np.ones(len(cluster_index))  # label these cells as '1'

    marker_index = np.where(gene_list == marker_gene)[0]
    marker_expression = expression_data[:, marker_index]
    gene_index = np.where((cluster_id == target_cluster_id) & (marker_expression > 0))[
        0
    ]  # find cells expressing marker_gene in target_cluster_id

    gene_coor = coordinate[
        gene_index, :
    ]  # get the spatial coordinate of cells expressing marker_gene in target_cluster_id
    gene_lbl = 2 * np.ones(len(gene_index))  # label these cells as '2'

    x = np.vstack((gene_coor, cluster_coor))
    cluster_id = np.hstack((gene_lbl, cluster_lbl))
    cluster_name = [
        f"{marker_gene} cluster {target_cluster_id} cells",
        f"All cluster {target_cluster_id} cells",
    ]

    assert x.shape[1] == 2, "We need 2D data."

    num_cluster = len(np.unique(cluster_id))

    # automatically set the cluster name if needed
    if cluster_name is None:
        cluster_name = ["cluster " + str(i) for i in range(1, num_cluster + 1)]

    # get the angle list with resolution of 1000
    resolution = 1000
    angle_list = np.pi * np.linspace(0, 360, resolution) / 180

    colors = ["red", "gray"]
    alphas = [1, 0.25]

    # for each cluster, get the list of gini corresponding to each angle
    for idx, cluster in enumerate(range(1, num_cluster + 1)):
        coordinate = x[cluster_id == cluster]
        gini = np.zeros(len(angle_list))

        for i, angle in enumerate(angle_list):
            value = np.dot(coordinate, [np.cos(angle), np.sin(angle)])
            value -= np.min(value)
            gini[i] = compute_gini(np.ones(len(value)), value)[0]

        # Polar plot with specified color
        plt.polar(angle_list, gini, color=colors[idx], alpha=alphas[idx])
        plt.title("Gini Coefficient")
        
    plt.legend(cluster_name)
    
    # Save figure in png format
    file_name = f"gini_{marker_gene}_c-{target_cluster_id}.png"
    file_path = os.path.join(tmp_dir, file_name)
    plt.savefig(file_path)
