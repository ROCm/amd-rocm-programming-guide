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
left_nav_title = f"AMD ROCm™ Programming Guide {version_number}"

# ROCm version numbers
rocm_version = '7.1.1'
rocm_major_version = '7.0'
rocm_multi_versions = '7.1.1 7.0.2' # in 6.3, the folder names on repo.radeon.com use 6.3 for minor releases
rocm_multi_versions_package_versions = '7.1.1 7.0.2' # however, in multi, the packages use 6.3.0
rocm_directory_version = '7.1.1' # in 6.0 rocm was located in /opt/rocm-6.0.0
amdgpu_version = '7.1.1' # directory in https://repo.radeon.com/rocm/apt/ and https://repo.radeon.com/amdgpu-install/
amdgpu_install_version = '7.1.1.70101-1' # version in https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/
udev_version = '30.20.1.0-2255209'
udev_amdgpu_version = '30.20.1'

# for PDF output on Read the Docs
project = "AMD ROCm™ Programming Guide"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number
latex_engine = "xelatex"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.setup()

external_projects_current_project = "amd-rocm-programming-guide"

# Add the following replacements to every RST file.
rst_prolog = f"""
.. |rocm_version| replace:: {rocm_version}
.. |rocm_major_version| replace:: {rocm_major_version}
.. |rocm_multi_versions| replace:: {rocm_multi_versions}
.. |rocm_multi_versions_package_versions| replace:: {rocm_multi_versions_package_versions}
.. |amdgpu_version| replace:: {amdgpu_version}
.. |rocm_directory_version| replace:: {rocm_directory_version}
.. |amdgpu_install_version| replace:: {amdgpu_install_version}
.. |udev_version| replace:: {udev_version}
.. |udev_amdgpu_version| replace:: {udev_amdgpu_version}
"""

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)

# Add the _extensions directory to Python's search path
sys.path.append(str(Path(__file__).parent / 'extension'))

extensions += ["sphinxcontrib.datatemplates", "version-ref", "csv-to-list-table", "remote-content", "svg-pdf-converter", "sphinx_subfigure", "sphinx_substitution_extensions"]

cpp_id_attributes = ["__global__", "__device__", "__host__", "__forceinline__", "static"]
cpp_paren_attributes = ["__declspec"]

suppress_warnings = ["etoc.toctree"]

# Check if the branch is a docs/ branch
official_branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.find("docs/")

# Supported linux version numbers
ubuntu_version_numbers = [('24.04', 'noble'), ('22.04', 'jammy')]
debian_version_numbers = [('13', 'noble'), ('12', 'jammy')]
debian_udev_versions = [('13', 'noble', '24.04'), ('12', 'jammy', '22.04')]
rhel_release_version_numbers = ['10', '9', '8']
rhel_version_numbers = ['10.1', '10.0', '9.7', '9.6', '9.4', '8.10']
rhel_multi_versions = ['10.0', '9.6', '9.4', '8.10']
sles_version_numbers = ['15.7']
ol_release_version_numbers = ['10', '9', '8']
ol_version_numbers = ['10.0', '9.6', '8.10']
azl_version_numbers = ['3.0']
rl_version_numbers = ['9.6']

html_context = {
    "ubuntu_version_numbers" : ubuntu_version_numbers,
    "debian_version_numbers" : debian_version_numbers,
    "debian_udev_versions" : debian_udev_versions,
    "sles_version_numbers" : sles_version_numbers,
    "rhel_release_version_numbers" : rhel_release_version_numbers,
    "rhel_version_numbers" : rhel_version_numbers,
    "rhel_multi_versions" : rhel_multi_versions,
    "ol_release_version_numbers" : ol_release_version_numbers,
    "ol_version_numbers" : ol_version_numbers,
    "azl_version_numbers": azl_version_numbers,
    "rl_version_numbers" : rl_version_numbers
}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_theme_options = {
    "announcement": "Additional content can be found on the <a id='rocm-banner' href='https://rocm.docs.amd.com/en/latest/'>ROCm™ documentation portal</a>.",
    "flavor": "generic",
    "header_title": "AMD ROCm™ Programming Guide",
    "header_link": "https://rocm.docs.amd.com/projects/amd-rocm-programming-guide/en/latest/",
    "version_list_link": "https://rocm.docs.amd.com/projects/amd-rocm-programming-guide/en/latest/release/versions.html",
    "nav_secondary_items": {
        "GitHub": "https://github.com/ROCm/amd-rocm-programming-guide",
        "Community": "https://github.com/ROCm/ROCm/discussions",
        "Blogs": "https://rocm.blogs.amd.com/",
        "ROCm™ Docs": "https://rocm.docs.amd.com",
        "Support": "https://github.com/ROCm/ROCm/issues/new/choose",
    },
    "link_main_doc": False,
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    }
}

html_context["official_branch"] = official_branch
html_context["version"] = version
html_context["release"] = release

html_theme = "rocm_docs_theme"

numfig = False

# SVG converter configuration is handled by the svg-pdf-converter extension
# which provides custom preprocessing for Draw.io SVGs
