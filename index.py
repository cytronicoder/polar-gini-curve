from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import os
from util.render_markdown import (
    render_markdown,
)

app = Flask(__name__)

# Create temporary storage for the plot
if not os.path.exists("/tmp"):
    os.mkdir("/tmp")


@app.route("/")
def index():
    return render_markdown("README.md", "home.html")


@app.route("/playground/", methods=["GET", "POST"])
def playground():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            without_extension = os.path.splitext(filename)[0]

            # TODO: use the file data and the functions from script to generate plot
            # Below is an example of creating a simple plot
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            plt.plot(x, y)

            # Save the plot as an image in temporary storage
            filename = without_extension + "_plot.png"
            plt.savefig(os.path.join("/tmp", filename))
            plt.close()

            # Pass image filename to the template to display it
            return render_template("display.html", filename=filename)

    return render_template("upload.html")


@app.route("/tmp/<path:filename>")
def serve_tmp_file(filename):
    return send_from_directory("/tmp", filename)


if __name__ == "__main__":
    app.run(debug=True)
