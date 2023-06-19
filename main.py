from flask import Flask
from util.render_markdown import render_markdown


app = Flask(__name__)

@app.route("/")
def index():
    return render_markdown("README.md")


if __name__ == "__main__":
    app.run(debug=True)
