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

from rocm_docs import ROCmDocs

version_numbers = []
version_file = open("../VERSION", "r")
lines = version_file.readlines()
for line in lines:
    if line[0] == '#':
        continue
    version_numbers.append(line.strip())
version_number = ".".join(version_numbers)
left_nav_title = f"HIP book {version_number} Documentation"

# for PDF output on Read the Docs
project = "HIP Book Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.setup()

external_projects_current_project = "hip"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Add the _extensions directory to Python's search path
sys.path.append(str(Path(__file__).parent / 'extension'))

extensions += ["sphinxcontrib.datatemplates", "version-ref", "csv-to-list-table", "remote-content"]

cpp_id_attributes = ["__global__", "__device__", "__host__", "__forceinline__", "static"]
cpp_paren_attributes = ["__declspec"]

suppress_warnings = ["etoc.toctree"]

# Check if the branch is a docs/ branch
official_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.find("docs/")

html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_context["official_branch"] = official_branch
html_context["version"] = version
html_context["release"] = release


numfig = False
suppress_warnings = ["autosectionlabel.*"]
