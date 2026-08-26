import urllib.request
import os

repo = "ROCm/ROCm"
branch = "docs/10.0.0"

def fetch(remote_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/{remote_path}",
        local_path
    )

# Main install page
fetch("docs/install/rocm.rst", "docs/install/rocm.rst")

# Include files
fetch("docs/install/includes/000-intro.rst",                          "docs/install/includes/000-intro.rst")
fetch("docs/install/includes/100-prerequisites.rst",                  "docs/install/includes/100-prerequisites.rst")
fetch("docs/install/includes/150-runfile-quick-start-config-options.rst", "docs/install/includes/150-runfile-quick-start-config-options.rst")
fetch("docs/install/includes/200-install.rst",                        "docs/install/includes/200-install.rst")
fetch("docs/install/includes/300-post-install.rst",                   "docs/install/includes/300-post-install.rst")
fetch("docs/install/includes/400-uninstall.rst",                      "docs/install/includes/400-uninstall.rst")
fetch("docs/install/includes/fam-multi-arch-selector.rst",            "docs/install/includes/fam-multi-arch-selector.rst")
fetch("docs/install/includes/fam-selector.rst",                       "docs/install/includes/fam-selector.rst")
fetch("docs/install/includes/gpu-selector.rst",                       "docs/install/includes/gpu-selector.rst")
fetch("docs/install/includes/install-method-selector.rst",            "docs/install/includes/install-method-selector.rst")
fetch("docs/install/includes/os-selector.rst",                        "docs/install/includes/os-selector.rst")
fetch("docs/install/includes/os-ver-selector.rst",                    "docs/install/includes/os-ver-selector.rst")

# Additional install pages
fetch("docs/install/rocm-packages.rst",    "docs/install/rocm-packages.rst")
fetch("docs/install/build-from-source.rst", "docs/install/build-from-source.rst")

# Redirect stubs (TOC targets for JS selector)
fetch("docs/install/redirect/_prerequisites.rst",  "docs/install/redirect/_prerequisites.rst")
fetch("docs/install/redirect/_install.rst",        "docs/install/redirect/_install.rst")
fetch("docs/install/redirect/_post-install.rst",   "docs/install/redirect/_post-install.rst")
fetch("docs/install/redirect/_uninstall.rst",      "docs/install/redirect/_uninstall.rst")
