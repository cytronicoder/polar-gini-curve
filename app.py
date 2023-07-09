"""
Flask server module for processing biological data and rendering interactive results.
"""

import tempfile
from io import BytesIO

import numpy as np
from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    url_for,
    session,
    send_from_directory,
)
from scipy.io import loadmat

from util.render_markdown import render_markdown
from util.polar_gini_curves import graph_tsne, graph_gini

app = Flask(__name__)
app.secret_key = "polarginicurves"  # dummy secret key

gene_list = set()
temp_dir = tempfile.mkdtemp()


@app.route("/")
def index():
    """Render the home page."""
    return render_markdown("README.md", "home.html")


@app.route("/playground/")
def playground():
    """Render the upload page."""
    return render_template("upload.html")


@app.route("/process_clusters", methods=["POST"])
def process_clusters():
    """
    Process the uploaded clusters, identify the unique clusters,
    and return them as a JSON response.
    """
    file = request.files["cluster_id"]
    if file:
        file_bytes = BytesIO(file.read())
        mat_content = loadmat(file_bytes)

        # Assuming the mat file has a variable named 'clusters' with cluster data
        clusters = mat_content.get("clusterID")
        if clusters is not None:
            unique_clusters = np.unique(clusters).tolist()
            unique_clusters[unique_clusters.index(-1)] = "None"
            return jsonify({"clusters": unique_clusters})


@app.route("/check_gene_marker", methods=["POST"])
def check_gene_marker():
    """Check if a gene marker exists in the gene list."""
    data = request.json
    gene_marker = data.get("gene_marker")

    # Here, check if gene_marker exists in gene_list
    exists = gene_marker in gene_list

    return jsonify({"exists": exists})


@app.route("/process_gene_list", methods=["POST"])
def process_gene_list():
    """Process the uploaded gene list."""
    if "gene_list" in request.files:
        file = request.files["gene_list"]
        if file.filename.endswith(".mat"):
            file_bytes = BytesIO(file.read())
            mat = loadmat(file_bytes)
            global gene_list

            flattened_gene_list = [item[0][0] for item in mat.get("geneList")]
            gene_list = set(flattened_gene_list)
    return (
        "",
        204,
    )


@app.route("/draw_tsne", methods=["POST"])
def draw_tsne():
    """Generate the t-SNE graph based on the uploaded data and inputs from the session."""
    gene_marker, selected_cluster = get_marker_and_cluster()
    coordinate, cluster_id, expression, gene_list = get_files_data()

    # Call the graph_tsne function to generate the graph
    graph_tsne(
        marker_gene=gene_marker,
        coordinate=coordinate,
        cluster_id=cluster_id,
        target_cluster_id=int(selected_cluster),
        expression_data=expression,
        gene_list=gene_list,
        random_state=0,
        tmp_dir=temp_dir,
    )

    # Render tsne_display.html
    return jsonify(url=url_for("display_tsne"))


@app.route("/generate_gini", methods=["POST"])
def generate_gini():
    """Generate the Gini graph based on the uploaded data and inputs from the session."""
    gene_marker, selected_cluster = get_marker_and_cluster()
    coordinate, cluster_id, expression, gene_list = get_files_data()

    # Call the graph_gini function to generate the graph
    graph_gini(
        marker_gene=gene_marker,
        coordinate=coordinate,
        cluster_id=cluster_id,
        target_cluster_id=int(selected_cluster),
        expression_data=expression,
        gene_list=gene_list,
        tmp_dir=temp_dir,
    )

    # Render gini_display.html
    return jsonify(url=url_for("display_gini"))


@app.route("/display_tsne")
def display_tsne():
    """Display the t-SNE graph."""
    gene_marker = session.get("gene_marker")
    selected_cluster = session.get("selected_cluster")
    return render_template(
        "display.html",
        filename=f"tsne_{gene_marker}_c-{selected_cluster}.png",
    )


@app.route("/display_gini")
def display_gini():
    """Display the Gini graph."""
    gene_marker = session.get("gene_marker")
    selected_cluster = session.get("selected_cluster")
    return render_template(
        "display.html",
        filename=f"gini_{gene_marker}_c-{selected_cluster}.png",
    )


@app.route("/tmp/<path:filename>")
def serve_tmp_file(filename):
    """Serve temporary files."""
    return send_from_directory(temp_dir, filename)


def get_marker_and_cluster():
    """Retrieve gene marker and selected cluster from session or form."""
    if "gene_marker" in session and "selected_cluster" in session:
        gene_marker = session["gene_marker"]
        selected_cluster = session["selected_cluster"]
    else:
        gene_marker = request.form["gene_marker"]
        selected_cluster = request.form["selected_cluster"]

        session["gene_marker"] = gene_marker
        session["selected_cluster"] = selected_cluster

    return gene_marker, selected_cluster


def get_files_data():
    """Load and parse data from uploaded files."""
    # Handle the uploaded files
    coordinate_file = request.files["coordinate"]
    cluster_id_file = request.files["cluster_id"]
    expression_file = request.files["expression"]
    gene_list_file = request.files["gene_list"]

    # Load the files
    coordinate_file = BytesIO(coordinate_file.read())
    cluster_id_file = BytesIO(cluster_id_file.read())
    expression_file = BytesIO(expression_file.read())
    gene_list_file = BytesIO(gene_list_file.read())

    # Read the data from the files (assuming they are .mat files)
    coordinate = loadmat(coordinate_file)["coordinate"]
    cluster_id = loadmat(cluster_id_file)["clusterID"]
    expression = loadmat(expression_file)["Expression"]
    gene_list = loadmat(gene_list_file)["geneList"]

    return coordinate, cluster_id, expression, gene_list


if __name__ == "__main__":
    app.run(debug=True)
