import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
from pygments.formatters import HtmlFormatter


def load_css(css_file):
    with open(css_file, "r") as f:
        return f.read()


def render_markdown(file_name, css_file="static/custom.css"):
    readme_file = open(file_name, "r", encoding="utf-8")

    md_template_string = markdown.markdown(
        readme_file.read(), extensions=["fenced_code", "codehilite"]
    )

    formatter = HtmlFormatter(style="emacs", full=True, cssclass="codehilite")
    css_string = formatter.get_style_defs()
    md_css_string = "<style>" + css_string + "</style>"

    custom_css = load_css(css_file)
    custom_css_string = "<style>" + custom_css + "</style>"

    md_template = md_css_string + custom_css_string + md_template_string
    return md_template
