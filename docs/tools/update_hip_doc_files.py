import os
import urllib.request
import urllib.error

repo = "ROCm/rocm-systems"
branch = "docs/10.0.0"

BASE = f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}"

# (source path relative to BASE, destination path in this repo)
FILES = [
    # hip_runtime_api.rst
    ("projects/hip/docs/how-to/hip_runtime_api.rst", "docs/how-to/hip_runtime_api.rst"),

    # hip_runtime
    ("projects/hip/docs/how-to/hip_runtime_api/asynchronous.rst", "docs/how-to/hip_runtime_api/asynchronous.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/call_stack.rst", "docs/how-to/hip_runtime_api/call_stack.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/cooperative_groups.rst", "docs/how-to/hip_runtime_api/cooperative_groups.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/error_handling.rst", "docs/how-to/hip_runtime_api/error_handling.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/execution_context.rst", "docs/how-to/hip_runtime_api/execution_context.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/external_interop.rst", "docs/how-to/hip_runtime_api/external_interop.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/hipgraph.rst", "docs/how-to/hip_runtime_api/hipgraph.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/initialization.rst", "docs/how-to/hip_runtime_api/initialization.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management.rst", "docs/how-to/hip_runtime_api/memory_management.rst"),

    # The multi device downloaded to a different location
    ("projects/hip/docs/how-to/hip_runtime_api/multi_device.rst", "docs/how-to/multi-gpu/multi_device.rst"),

    ("projects/hip/docs/how-to/hip_runtime_api/opengl_interop.rst", "docs/how-to/hip_runtime_api/opengl_interop.rst"),

    # hip_runtime / memory_management
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/virtual_memory.rst", "docs/how-to/hip_runtime_api/memory_management/virtual_memory.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/coherence_control.rst", "docs/how-to/hip_runtime_api/memory_management/coherence_control.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/device_memory.rst", "docs/how-to/hip_runtime_api/memory_management/device_memory.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/host_memory.rst", "docs/how-to/hip_runtime_api/memory_management/host_memory.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.rst", "docs/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.rst"),
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/unified_memory.rst", "docs/how-to/hip_runtime_api/memory_management/unified_memory.rst"),

    # hip_runtime / memory_management / device_memory
    ("projects/hip/docs/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.rst", "docs/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.rst"),

    # tutorial
    ("projects/hip/docs/tutorial/reduction.rst", "docs/tutorial/hip-performance-optimization/reduction.rst"),
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
