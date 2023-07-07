import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
from pygments.formatters import HtmlFormatter
from flask import render_template
import html


def load_css(css_file):
    with open(css_file, "r") as f:
        return f.read()


def render_markdown(file_name, template_name):
    readme_file = open(file_name, "r", encoding="utf-8")

    md_template_string = markdown.markdown(
        readme_file.read(), extensions=["fenced_code", "codehilite"]
    )

    formatter = HtmlFormatter(style="emacs", full=True, cssclass="codehilite")
    css_string = formatter.get_style_defs()
    md_css_string = "<style>" + css_string + "</style>"

    # final_css = md_css_string + custom_css_string
    # md_template_string = html.unescape(md_template_string)
    # final_css = html.unescape(final_css)

    md_template = md_css_string + md_template_string

    return render_template(
        template_name, content=md_template
    )
