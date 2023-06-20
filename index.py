from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import os
from util.render_markdown import (
    render_markdown,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"


@app.route("/")
def index():
    return render_markdown("README.md")


@app.route("/playground/", methods=["GET", "POST"])
def playground():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            # TODO: use the file data and the functions from script to generate plot and save it to a file

            return render_template("display.html")

    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
