import urllib.error
import urllib.request
import os
import subprocess
import sys

repo = "ROCm/ROCm"
branch = "docs/7.14.1"

BASE = f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}"

# (source path relative to BASE, destination path in this repo)
# The install docs mirror the upstream ROCm layout: include/ (singular) with
# inline .rst selector fragments (os-selector, per-OS version selectors, etc.).
FILES = [
    # Main install page
    ("docs/install/rocm.rst", "docs/install/rocm.rst"),

    # Additional install pages
    ("docs/install/build-from-source.rst", "docs/install/build-from-source.rst"),

    # Include files (numbered sections + pip table)
    ("docs/install/include/000-intro.rst", "docs/install/include/000-intro.rst"),
    ("docs/install/include/050-install-methods.rst", "docs/install/include/050-install-methods.rst"),
    ("docs/install/include/100-prerequisites.rst", "docs/install/include/100-prerequisites.rst"),
    ("docs/install/include/150-runfile-quick-start-config-options.rst", "docs/install/include/150-runfile-quick-start-config-options.rst"),
    ("docs/install/include/200-install.rst", "docs/install/include/200-install.rst"),
    ("docs/install/include/300-post-install.rst", "docs/install/include/300-post-install.rst"),
    ("docs/install/include/400-uninstall.rst", "docs/install/include/400-uninstall.rst"),
    ("docs/install/include/pip-packages-table.rst", "docs/install/include/pip-packages-table.rst"),

    # Selector fragments (family/GPU/OS/OS-version)
    ("docs/install/include/fam-multi-arch-selector.rst", "docs/install/include/fam-multi-arch-selector.rst"),
    ("docs/install/include/gpu-selector.rst", "docs/install/include/gpu-selector.rst"),
    ("docs/install/include/ror-gpu-selector.rst", "docs/install/include/ror-gpu-selector.rst"),
    ("docs/install/include/install-method-selector.rst", "docs/install/include/install-method-selector.rst"),
    ("docs/install/include/os-selector.rst", "docs/install/include/os-selector.rst"),
    ("docs/install/include/debian-ver-selector.rst", "docs/install/include/debian-ver-selector.rst"),
    ("docs/install/include/oracle-linux-ver-selector.rst", "docs/install/include/oracle-linux-ver-selector.rst"),
    ("docs/install/include/rhel-ver-selector.rst", "docs/install/include/rhel-ver-selector.rst"),
    ("docs/install/include/rocky-linux-ver-selector.rst", "docs/install/include/rocky-linux-ver-selector.rst"),
    ("docs/install/include/sles-ver-selector.rst", "docs/install/include/sles-ver-selector.rst"),
    ("docs/install/include/ubuntu-ver-selector.rst", "docs/install/include/ubuntu-ver-selector.rst"),
    ("docs/install/include/windows-ver-selector.rst", "docs/install/include/windows-ver-selector.rst"),

    # Redirect stubs (TOC targets for JS selector)
    ("docs/install/redirect/_prerequisites.rst", "docs/install/redirect/_prerequisites.rst"),
    ("docs/install/redirect/_install-methods.rst", "docs/install/redirect/_install-methods.rst"),
    ("docs/install/redirect/_install.rst", "docs/install/redirect/_install.rst"),
    ("docs/install/redirect/_post-install.rst", "docs/install/redirect/_post-install.rst"),
    ("docs/install/redirect/_uninstall.rst", "docs/install/redirect/_uninstall.rst"),
]


# Post-fetch fixups: (path relative to this repo, search text, replacement).
# Upstream ROCm keeps 200-install.rst directly under docs/install/, so its
# nested ``.. include:: include/pip-packages-table.rst`` resolves correctly
# there. In this repo the file lives under docs/install/include/, so that
# relative path would resolve to install/include/include/ and fail to build.
# Rewrite it to an absolute source-root path (leading slash) that is correct
# regardless of where the including file sits. Re-applied on every fetch
# because each download overwrites the file with the upstream relative form.
FIXUPS = [
    (
        "docs/install/include/200-install.rst",
        ".. include:: include/pip-packages-table.rst",
        ".. include:: /install/include/pip-packages-table.rst",
    ),
]


# Top-level pages that must carry a ``:robots: noindex`` meta block. The fetch
# overwrites the local noindex with upstream :description:/:keywords: meta, so
# restore it after each download by running add_noindex_meta.py (idempotent).
NOINDEX_PAGES = [
    "docs/install/rocm.rst",
    "docs/install/build-from-source.rst",
]


def apply_fixups() -> None:
    for path, search, replace in FIXUPS:
        with open(path, encoding="utf-8") as handle:
            content = handle.read()
        if search in content:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content.replace(search, replace))
            print(f"FIX  {path}: rewrote include path to source-root form")

    noindex_script = os.path.join(os.path.dirname(__file__), "add_noindex_meta.py")
    for page in NOINDEX_PAGES:
        subprocess.run([sys.executable, noindex_script, page], check=True)
        print(f"FIX  {page}: ensured :robots: noindex meta")


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

    apply_fixups()

    print(f"\n{len(FILES) - len(failures)}/{len(FILES)} downloaded.")
    if failures:
        print(f"{len(failures)} failed:")
        for dest, url in failures:
            print(f"  - {dest}: {url}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
