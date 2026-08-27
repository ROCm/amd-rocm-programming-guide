# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import os
import sys
from pathlib import Path
from subprocess import run
from pathlib import Path
from typing import Any, Dict, List

ROCM_VERSION = "10.0.0"
GA_DATE = "2026-08-26"

# for PDF output on Read the Docs
project = "AMD ROCm™ Programming Guide"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved."
version = ROCM_VERSION
release = ROCM_VERSION
latex_engine = "xelatex"

external_toc_path = "./sphinx/_toc.yml"

external_projects_current_project = "amd-rocm-programming-guide"

rocm_docs_pdf_mock_selector_state = {
    "install/rocm": [
        {"fam": "all", "w": "compute", "os": "ubuntu", "ubuntu-ver": "26.04"},
        {"fam": "all", "w": "compute", "os": "rhel", "rhel-ver": "10.2"},
        {"fam": "all", "w": "compute", "os": "windows"},
    ],
}

# Generate llms.txt and llms-full.txt (requires the rocm-docs-core[llms] extra).
rocm_docs_generate_llms = True

# For substitutions in MyST Markdown and rST files.
# Usage:
#   ```md              | ```rst
#   {{ ROCM_VERSION }} | |ROCM_VERSION|
#   ```                | ```
myst_substitutions = {"ROCM_VERSION": ROCM_VERSION}
rst_prolog = "\n".join(
    f".. |{key}| replace:: {val}" for key, val in myst_substitutions.items()
)

# Add the _extensions directory to Python's search path
sys.path.append(str(Path(__file__).parent / 'extension'))

extensions = ["rocm_docs", "rocm_docs.selector", "sphinxcontrib.datatemplates", "version-ref", "csv-to-list-table", "remote-content", "svg-pdf-converter", "sphinx_subfigure", "sphinx_substitution_extensions", "matrix"]

# Jinja templates consumed by the datatemplate:yaml selector directives on the
# install page (fam/gpu/os/os-version selectors generated from data/*.yaml).
templates_path = ["install/include/templates"]

cpp_id_attributes = ["__global__", "__device__", "__host__", "__forceinline__", "static"]
cpp_paren_attributes = ["__declspec"]

suppress_warnings = ["etoc.toctree", "etoc.ref", "toc.excluded"]

# Check if the branch is a docs/ branch
official_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.find("docs/")
html_context = {}

if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_theme_options = {
    "announcement": "Additional content can be found on the <a id='rocm-banner' href='https://rocm.docs.amd.com/en/latest/'>ROCm documentation portal</a>.",
    "flavor": "generic",
    "use_download_button": True,
    "header_title": "AMD ROCm™ Programming Guide",
    "header_link": "https://rocm-handbook.amd.com/projects/amd-rocm-programming-guide/en/docs-10.0.0/",
    "version_list_link": False,
    "nav_secondary_items": {
        "GitHub": "https://github.com/ROCm/amd-rocm-programming-guide",
        "Community": "https://github.com/ROCm/ROCm/discussions",
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm™ Docs": "https://rocm.docs.amd.com",
        "Instinct™ Docs": "https://instinct.docs.amd.com/",
        "Support": "https://github.com/ROCm/ROCm/issues/new/choose",
        "ROCm Developer Hub": "https://www.amd.com/en/developer/resources/rocm-hub.html",
    },
    "link_main_doc": False,
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    }
}

html_context["official_branch"] = official_branch
html_context["version"] = version
html_context["release"] = release

html_static_path = ["sphinx/static/css"]
html_css_files = ["rocm_custom.css"]

html_theme = "rocm_docs_theme"

numfig = False

# Exclude HTML-only redirect stub pages from PDF output.  These pages exist
# solely to give the sidebar JS navigation something to link to; they have no
# readable content.  The external-toc still references them; the "etoc.ref" and
# "toc.excluded" entries in suppress_warnings silence the resulting warnings.
rocm_docs_pdf_exclude_patterns = ["install/redirect/*"]

# SVG converter configuration is handled by the svg-pdf-converter extension
# which provides custom preprocessing for Draw.io SVGs
