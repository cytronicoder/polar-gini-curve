from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    url_for,
    session,
    send_from_directory,
)
from io import BytesIO
from scipy.io import loadmat

import numpy as np
import os
from util.render_markdown import (
    render_markdown,
)

from util.graph_tsne import graph_tsne

app = Flask(__name__)
app.secret_key = "polarginicurves"  # dummy secret key

# Create temporary storage for the plot
if not os.path.exists("/tmp"):
    os.mkdir("/tmp")

gene_list = set()


@app.route("/")
def index():
    return render_markdown("README.md", "home.html")


@app.route("/playground/")
def playground():
    return render_template("upload.html")


@app.route("/process_clusters", methods=["POST"])
def process_clusters():
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
    data = request.json
    gene_marker = data.get("gene_marker")

    # Here, check if gene_marker exists in gene_list
    exists = gene_marker in gene_list

    print(exists)

    return jsonify({"exists": exists})


@app.route("/process_gene_list", methods=["POST"])
def process_gene_list():
    if "gene_list" in request.files:
        file = request.files["gene_list"]
        if file.filename.endswith(".mat"):
            file_bytes = BytesIO(file.read())
            mat = loadmat(file_bytes)
            global gene_list

            flattened_gene_list = [item[0][0] for item in mat.get("geneList")]
            gene_list = set(flattened_gene_list)
            print(gene_list)
    return (
        "",
        204,
    )


@app.route("/draw_tsne", methods=["POST"])
def draw_tsne():
    gene_marker = request.form["gene_marker"]
    selected_cluster = request.form["selected_cluster"]

    session["gene_marker"] = gene_marker
    session["selected_cluster"] = selected_cluster

    # Handle the uploaded files
    coordinate_file = request.files["coordinate"]
    cluster_id_file = request.files["cluster_id"]
    expression_file = request.files["expression"]
    gene_list = request.files["gene_list"]

    # Load the files
    coordinate_file = BytesIO(coordinate_file.read())
    cluster_id_file = BytesIO(cluster_id_file.read())
    expression_file = BytesIO(expression_file.read())
    gene_list = BytesIO(gene_list.read())

    # Read the data from the files (assuming they are .mat files)
    coordinate = loadmat(coordinate_file)["coordinate"]
    cluster_id = loadmat(cluster_id_file)["clusterID"]
    expression = loadmat(expression_file)["Expression"]
    gene_list = loadmat(gene_list)["geneList"]

    # Call the draw_tsne function to generate the graph
    graph_tsne(
        marker_gene=gene_marker,
        coordinate=coordinate,
        cluster_id=cluster_id,
        target_cluster_id=int(selected_cluster),
        expression_data=expression,
        gene_list=gene_list,  # This should be loaded previously or you can include it in the form
        random_state=0,
    )

    # Render display.html
    return jsonify(url=url_for("display_tsne"))


@app.route("/display_tsne")
def display_tsne():
    gene_marker = session.get("gene_marker")
    selected_cluster = session.get("selected_cluster")
    return render_template(
        "display.html", title_for=f"t-SNE for marker gene {gene_marker} in cluster {selected_cluster}", filename=f"tsne_{gene_marker}_c-{selected_cluster}.png"
    )


@app.route("/tmp/<path:filename>")
def serve_tmp_file(filename):
    return send_from_directory("/tmp", filename)


if __name__ == "__main__":
    app.run(debug=True)
