"""
This module provides utility functions for rendering Markdown content.
"""

from flask import render_template
from pygments.formatters import HtmlFormatter
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import markdown


def load_css(css_file):
    """
    Function to read and return the contents of a CSS file.

    Args:
        css_file (str): The path to the CSS file.

    Returns:
        str: The contents of the CSS file.
    """
    with open(css_file, "r", encoding="utf-8") as file:
        return file.read()


def render_markdown(file_name, template_name):
    """
    Function to convert a Markdown file to HTML, add syntax highlighting, and render it in a template.

    Args:
        file_name (str): The path to the Markdown file.
        template_name (str): The name of the Flask template to render.

    Returns:
        str: The rendered template with the converted Markdown content.
    """
    with open(file_name, "r", encoding="utf-8") as readme_file:
        md_content = readme_file.read()

    md_template_string = markdown.markdown(
        md_content, extensions=["fenced_code", "codehilite"]
    )

    formatter = HtmlFormatter(style="emacs", full=True, cssclass="codehilite")
    css_string = formatter.get_style_defs()
    md_css_string = f"<style>{css_string}</style>"

    md_template = md_css_string + md_template_string

    return render_template(template_name, content=md_template)
