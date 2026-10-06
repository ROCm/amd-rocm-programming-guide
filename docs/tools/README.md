# Documentation tooling

This directory holds the scripts that assemble a published version of the AMD
ROCm Programming Guide. The guide vendors source files (how-to pages, install
pages, data-center integration docs, and example code) from several upstream
ROCm repositories, pinned to a release's branches. These scripts fetch that
content and prepare it for publishing.

## Scripts

| Script | Fetches | From |
|--------|---------|------|
| `update_hip_doc_files.py` | HIP runtime API how-to pages (`docs/how-to/`) | `ROCm/rocm-systems` at `docs/<version>` |
| `update_rocm_install_doc_files.py` | Install pages + selector data (`docs/install/`, `docs/data/`) | `ROCm/ROCm` at `docs/<version>` |
| `update_datacenters_doc_files.py` | Slurm + Kubernetes integration pages (`docs/reference/rocm_in_data_centers/`) | `device-metrics-exporter`, `k8s-device-plugin` |
| `update_example_codes.py` | HIP example sources (`docs/tools/example_codes/`) | `ROCm/rocm-examples` at `release/therock-<major.minor>` |
| `add_noindex_meta.py` | — (post-processing) | adds the `noindex` meta tag to fetched pages |

> **Run every script from the repository root**, not from `docs/tools/`. The
> fetch scripts write to destination paths relative to the current directory
> (e.g. `docs/how-to/...`). Running them from `docs/tools/` creates a stray
> `docs/tools/docs/` tree instead.

## Creating a new published version (`docs/<version>`)

The guide publishes one branch per ROCm release (e.g. `docs/10.0.0`,
`docs/10.1.0`). Each is cut from `develop`, then bumped and re-fetched.

> **Base on `develop` of the public repo** (`ROCm/amd-rocm-programming-guide`,
> the `public` remote), not the internal repo's `develop`. The public
> `develop` is the active trunk and already carries the current `conf.py`
> version structure. The internal repo's `develop` can lag far behind.

### 1. Branch from develop

```sh
git fetch public
git checkout -b docs/<version> public/develop
```

### 2. Bump the version and repoint the fetch scripts

Edit `docs/conf.py`:

- `ROCM_VERSION` → `<version>` (e.g. `10.1.0`)
- `GA_DATE` → the release's GA date (`YYYY-MM-DD`)
- `header_link` → `.../en/docs-<version>/`

Edit the fetch scripts to point at the release's upstream branches:

- `update_hip_doc_files.py`: `branch = "docs/<version>"`
- `update_rocm_install_doc_files.py`: `branch = "docs/<version>"`
- `update_example_codes.py`: `BASE = ".../release/therock-<major.minor>"`

`update_example_codes.py` also has an `AMD_STAGING` override mechanism for
examples that have not yet synced from `amd-staging` to the release branch.
Before relying on it, check whether each overridden example now exists on the
release branch and drop the override if so:

```sh
curl -sI "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/release/therock-<major.minor>/<path>" | head -1
```

### 3. Run the fetch scripts (from the repo root)

On the AMD network, `urllib` needs the Zscaler root CA. Export it first (see
the repository's build notes for creating `combined-ca.pem`):

```sh
export SSL_CERT_FILE=$HOME/combined-ca.pem
export REQUESTS_CA_BUNDLE=$HOME/combined-ca.pem

python docs/tools/update_hip_doc_files.py
python docs/tools/update_rocm_install_doc_files.py
python docs/tools/update_datacenters_doc_files.py
python docs/tools/update_example_codes.py
```

Each prints a per-file `OK`/`FAIL` line and a final count. A single `FAIL` is
often a transient network error — rerun the script (it is idempotent) before
investigating a genuinely missing upstream path.

### 4. Re-add the `noindex` meta tag

The fetch overwrites each page with the upstream copy, which does **not**
carry the `noindex` meta tag the staged public mirror needs (it keeps the
in-progress docs out of search engines). Re-add it to the fetched pages:

```sh
python docs/tools/add_noindex_meta.py docs \
    --exceptions docs/tools/noindex_exceptions.txt
```

The script adds `.. meta:: :robots: noindex` to RST files and the equivalent
`myst.html_meta` frontmatter to Markdown files, skipping anything already
tagged or listed in `noindex_exceptions.txt`.

### 5. Build and commit

Build the docs locally to confirm the fetched content is valid:

```sh
python -m sphinx -b html docs docs/_build/html
```

Intersphinx `objects.inv` 404 warnings are expected in a local build (they
resolve against other published projects) and are not content errors.

Commit the work in logical groups — the version bump separately from each
script's fetched output — then open a PR against the public `docs/<version>`
branch.
