import urllib.request

repo = "ROCm/hip"
branch = "docs/develop"
# repo = "ROCm/rocm-systems"
# branch = "develop/projects/hip"

# "https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/develop/projects/hip/docs/how-to/hip_runtime_api.rst",
# https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/release/rocm-rel-7.0/projects/hip/docs/how-to/hip_runtime_api.rst

# Update update_example_codes.py

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/tools/update_example_codes.py",
     "docs/tools/update_example_codes.py"
)

# hip_runtime.rst

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api.rst",
     "docs/how-to/hip_runtime_api.rst"
)

# hip_runtime

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/asynchronous.rst",
     "docs/how-to/hip_runtime_api/asynchronous.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/call_stack.rst",
     "docs/how-to/hip_runtime_api/call_stack.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/cooperative_groups.rst",
     "docs/how-to/hip_runtime_api/cooperative_groups.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/error_handling.rst",
     "docs/how-to/hip_runtime_api/error_handling.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/external_interop.rst",
     "docs/how-to/hip_runtime_api/external_interop.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/hipgraph.rst",
     "docs/how-to/hip_runtime_api/hipgraph.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/initialization.rst",
     "docs/how-to/hip_runtime_api/initialization.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management.rst",
     "docs/how-to/hip_runtime_api/memory_management.rst"
)

# The multi device downloaded to a different location
urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/multi_device.rst",
     "docs/how-to/multi-gpu/multi_device.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/opengl_interop.rst",
     "docs/how-to/hip_runtime_api/opengl_interop.rst"
)

# hip_runtime / memory_management

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/virtual_memory.rst",
     "docs/how-to/hip_runtime_api/memory_management/virtual_memory.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/coherence_control.rst",
     "docs/how-to/hip_runtime_api/memory_management/coherence_control.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/device_memory.rst",
     "docs/how-to/hip_runtime_api/memory_management/device_memory.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/host_memory.rst",
     "docs/how-to/hip_runtime_api/memory_management/host_memory.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.rst",
     "docs/how-to/hip_runtime_api/memory_management/stream_ordered_allocator.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/unified_memory.rst",
     "docs/how-to/hip_runtime_api/memory_management/unified_memory.rst"
)

# hip_runtime / memory_management / device_memory

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.rst",
     "docs/how-to/hip_runtime_api/memory_management/device_memory/texture_fetching.rst"
)

# tutorial

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/tutorial/reduction.rst",
     "docs/tutorial/reduction.rst"
)

