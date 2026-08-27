import urllib.request
import urllib.error
import os

repo = "ROCm/ROCm"
branch = "docs/10.0.0"

BASE = f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}"

# (source path relative to BASE, destination path in this repo)
# The install docs mirror the upstream ROCm layout: include/ (singular) with the
# selector templates centralized under include/templates/ (consumed by the
# rocm-docs-core selector directive).
FILES = [
    # Main install page
    ("docs/install/rocm.rst", "docs/install/rocm.rst"),

    # Additional install pages
    ("docs/install/build-from-source.rst", "docs/install/build-from-source.rst"),

    # Include files (numbered sections + pip table)
    ("docs/install/include/000-install-methods.rst", "docs/install/include/000-install-methods.rst"),
    ("docs/install/include/050-intro.rst", "docs/install/include/050-intro.rst"),
    ("docs/install/include/100-prerequisites.rst", "docs/install/include/100-prerequisites.rst"),
    ("docs/install/include/150-runfile-quick-start-config-options.rst", "docs/install/include/150-runfile-quick-start-config-options.rst"),
    ("docs/install/include/200-install.rst", "docs/install/include/200-install.rst"),
    ("docs/install/include/300-post-install.rst", "docs/install/include/300-post-install.rst"),
    ("docs/install/include/400-uninstall.rst", "docs/install/include/400-uninstall.rst"),
    ("docs/install/include/pip-packages-table.rst", "docs/install/include/pip-packages-table.rst"),

    # Selector templates (rocm-docs-core selector directive)
    ("docs/install/include/templates/fam-selector.rst.jinja", "docs/install/include/templates/fam-selector.rst.jinja"),
    ("docs/install/include/templates/gpu-selector.rst.jinja", "docs/install/include/templates/gpu-selector.rst.jinja"),
    ("docs/install/include/templates/install-method-selector.rst.jinja", "docs/install/include/templates/install-method-selector.rst.jinja"),
    ("docs/install/include/templates/os-selector.rst.jinja", "docs/install/include/templates/os-selector.rst.jinja"),
    ("docs/install/include/templates/os-version-selector.rst.jinja", "docs/install/include/templates/os-version-selector.rst.jinja"),
    ("docs/install/include/templates/misc/os-selector-graphics-workloads.rst.jinja", "docs/install/include/templates/misc/os-selector-graphics-workloads.rst.jinja"),
    ("docs/install/include/templates/misc/os-selector-no-windows.rst.jinja", "docs/install/include/templates/misc/os-selector-no-windows.rst.jinja"),

    # Redirect stubs (TOC targets for JS selector)
    ("docs/install/redirect/_prerequisites.rst", "docs/install/redirect/_prerequisites.rst"),
    ("docs/install/redirect/_install.rst", "docs/install/redirect/_install.rst"),
    ("docs/install/redirect/_post-install.rst", "docs/install/redirect/_post-install.rst"),
    ("docs/install/redirect/_uninstall.rst", "docs/install/redirect/_uninstall.rst"),
]


def main() -> int:
    failures = []
    for src, dest in FILES:
        url = f"{BASE}/{src}"
        try:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            urllib.request.urlretrieve(url, dest)
            print(f"OK   {dest}")
        except urllib.error.HTTPError as err:
            print(f"FAIL {dest}: HTTP {err.code} {url}")
            failures.append((dest, url))
        except urllib.error.URLError as err:
            print(f"FAIL {dest}: {err.reason} {url}")
            failures.append((dest, url))

    print(f"\n{len(FILES) - len(failures)}/{len(FILES)} downloaded.")
    if failures:
        print(f"{len(failures)} failed:")
        for dest, url in failures:
            print(f"  - {dest}: {url}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
