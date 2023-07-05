from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import os
from util.render_markdown import (
    render_markdown,
)

app = Flask(__name__)

# check if folders exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists("static/images"):
    os.makedirs("static/images")

app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["IMAGE_FOLDER"] = "static/images/"


@app.route("/")
def index():
    return render_markdown("README.md", "home.html")


@app.route("/playground/", methods=["GET", "POST"])
def playground():
    if request.method == "POST":
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            without_extension = os.path.splitext(filename)[0]

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            print(f"{filename} uploaded!")

            # TODO: use the file data and the functions from script to generate plot
            # Below is an example of creating a simple plot
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            plt.plot(x, y)

            # Save the plot as an image in the "static/images" folder
            filename = without_extension + "_plot.png"
            plt.savefig(os.path.join(app.config['IMAGE_FOLDER'], filename))
            plt.close()

            # Pass image filename to the template to display it
            return render_template("display.html", filename=filename)

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
