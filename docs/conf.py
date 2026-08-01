"""Sphinx configuration for the soccer-vision docs.

Prose lives in Markdown (MyST) rather than reStructuredText so that a
contributor who only ever edits the README can edit these pages too.
"""

project = "soccer-vision"
copyright = "2026, Ben Weinstein"
author = "Ben Weinstein"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
]

exclude_patterns = ["_build"]

# `colon_fence` gives us ::: directives, which read better than a raw-RST
# block dropped into the middle of a Markdown file.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

html_theme = "furo"
html_title = "soccer-vision"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "source_repository": "https://github.com/bw4sz/soccer-video-analysis/",
    "source_branch": "master",
    "source_directory": "docs/",
}

# Don't copy the shell prompt when the reader clicks "copy".
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
