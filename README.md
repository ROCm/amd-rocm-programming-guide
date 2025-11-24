# HIP Book

The HIP Book is a solution designed for generating comprehensive books and
guides from the ROCm documentation portal. This repository primarily contains
HIP and ROCm documentation, and serves as a centralized resource for developers,
contributors, and users of the ROCm ecosystem.

The purpose of this repository is to:

- Aggregate and organize HIP and ROCm documentation into a structured book
  format.

- Provide an easily accessible and offline-readable version of the ROCm
  documentation.

- Support automated documentation generation workflows and publishing tools.

> [!NOTE]
> This repository focuses exclusively on documentation content and related build scripts — it does not contain any source code for HIP or ROCm themselves.

## Build the documentat

You can build our documentation via the command line using Python.

See the `build.tools.python` setting in the [Read the Docs configuration file](https://github.com/ROCm/ROCm/blob/develop/.readthedocs.yaml) for the Python version used by Read the Docs to build documentation.

See the [Python requirements file](https://github.com/ROCm/ROCm/blob/develop/docs/sphinx/requirements.txt) for Python packages needed to build the documentation.

Use the Python Virtual Environment (`venv`) and run the following commands from the project root:

::::{tab-set}
:::{tab-item} Linux and WSL
:sync: linux

```sh
python3 -mvenv .venv

.venv/bin/python -m pip install -r docs/sphinx/requirements.txt
.venv/bin/python -m sphinx -T -E -b html -d _build/doctrees -D language=en docs _build/html
```

:::
:::{tab-item} Windows
:sync: windows

```powershell
python -mvenv .venv

.venv\Scripts\python.exe -m pip install -r docs/sphinx/requirements.txt
.venv\Scripts\python.exe -m sphinx -T -E -b html -d _build/doctrees -D language=en docs _build/html
```

:::
::::

Navigate to `_build/html/index.html` and open this file in a web browser.

For further information, please check [building documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).
