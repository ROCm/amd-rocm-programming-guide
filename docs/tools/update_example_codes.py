# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import urllib.request
import urllib.error

BASE = "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/release/therock-10.0"
# Temporary override for examples not yet synced to release/therock-10.0.
AMD_STAGING = "https://raw.githubusercontent.com/ROCm/rocm-examples/refs/heads/amd-staging"
DEST = "docs/tools/example_codes"

# (source path relative to BASE, destination filename in DEST)
EXAMPLES = [
    # HIP-Basic
    ("HIP-Basic/opengl_interop/main.hip", "opengl_interop.hip"),
    ("HIP-Basic/vulkan_interop/main.hip", "external_interop.hip"),
    # Not yet on release/therock-10.0; fetch from amd-staging until it syncs.
    ("HIP-Basic/execution_context/main.hip", "execution_context.hip", AMD_STAGING),
    # Not yet on release/therock-10.0; fetch from amd-staging until it syncs (rocm-examples PR #486).
    ("HIP-Basic/cooperative_groups_double_buffered_tile/main.hip", "cooperative_groups_double_buffered_tile.hip", AMD_STAGING),
    ("HIP-Basic/cooperative_groups_prefix_sum/main.hip", "cooperative_groups_prefix_sum.hip", AMD_STAGING),

    # HIP-C++-Language-Extensions
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/calling_global_functions/main.hip", "calling_global_functions.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/extern_shared_memory/main.hip", "extern_shared_memory.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/launch_bounds/main.hip", "launch_bounds.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/set_constant_memory/main.hip", "set_constant_memory.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/template_warp_size_reduction/main.hip", "template_warp_size_reduction.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/timer/main.hip", "timer.hip"),
    ("HIP-Doc/Programming-Guide/HIP-C%2B%2B-Language-Extensions/warp_size_reduction/main.hip", "warp_size_reduction.hip"),

    # Porting-CUDA-code-to-HIP (formerly HIP-Porting-Guide)
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/device_code_feature_identification/main.hip", "device_code_feature_identification.hip"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/host_code_feature_identification/main.cpp", "host_code_feature_identification.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/identifying_compilation_target_platform/main.cpp", "identifying_compilation_target_platform.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/identifying_host_device_compilation_pass/main.hip", "identifying_host_device_compilation_pass.hip"),

    # Introduction-to-the-HIP-Programming-Model
    ("HIP-Doc/Programming-Guide/Introduction-to-the-HIP-Programming-Model/add_kernel/main.hip", "add_kernel.hip"),

    # Porting-CUDA-Driver-API (moved under Porting-CUDA-code-to-HIP)
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module/main.cpp", "load_module.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module_ex/main.cpp", "load_module_ex.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/load_module_ex_cuda/main.cpp", "load_module_ex_cuda.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/per_thread_default_stream/main.cpp", "per_thread_default_stream.cpp"),
    ("HIP-Doc/Programming-Guide/Porting-CUDA-code-to-HIP/pointer_memory_type/main.cpp", "pointer_memory_type.cpp"),

    # Programming-for-HIP-Runtime-Compiler
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/compilation_apis/main.cpp", "compilation_apis.cpp"),
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis/main.cpp", "linker_apis.cpp"),
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_file/main.cpp", "linker_apis_file.cpp"),
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/linker_apis_options/main.cpp", "linker_apis_options.cpp"),
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/lowered_names/main.cpp", "lowered_names.cpp"),
    ("HIP-Doc/Programming-Guide/Programming-for-HIP-Runtime-Compiler/rtc_error_handling/main.cpp", "rtc_error_handling.cpp"),

    # Using-HIP-Runtime-API / Asynchronous-Concurrent-Execution
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/async_kernel_execution/main.hip", "async_kernel_execution.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/event_based_synchronization/main.hip", "event_based_synchronization.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Asynchronous-Concurrent-Execution/sequential_kernel_execution/main.hip", "sequential_kernel_execution.hip"),

    # Using-HIP-Runtime-API / Call-Stack
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/call_stack_management/main.cpp", "call_stack_management.cpp"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Call-Stack/device_recursion/main.hip", "device_recursion.hip"),

    # Using-HIP-Runtime-API / Error-Handling
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Error-Handling/error_handling/main.hip", "error_handling.hip"),

    # Using-HIP-Runtime-API / HIP-Graphs
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_capture/main.hip", "graph_capture.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/HIP-Graphs/graph_creation/main.hip", "graph_creation.hip"),

    # Using-HIP-Runtime-API / Initialization
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Initialization/simple_device_query/main.cpp", "simple_device_query.cpp"),

    # Using-HIP-Runtime-API / Memory-Management / Device-Memory
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/constant_memory/main.hip", "constant_memory_device.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/dynamic_shared_memory/main.hip", "dynamic_shared_memory_device.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/explicit_copy/main.cpp", "explicit_copy.cpp"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/kernel_memory_allocation/main.hip", "kernel_memory_allocation.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Device-Memory/static_shared_memory/main.hip", "static_shared_memory_device.hip"),

    # Using-HIP-Runtime-API / Memory-Management / Host-Memory
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pageable_host_memory/main.cpp", "pageable_host_memory.cpp"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Host-Memory/pinned_host_memory/main.cpp", "pinned_host_memory.cpp"),

    # Using-HIP-Runtime-API / Memory-Management / SOMA
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/stream_ordered_memory_allocation/main.hip", "stream_ordered_memory_allocation.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/ordinary_memory_allocation/main.hip", "ordinary_memory_allocation.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool/main.hip", "memory_pool.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_resource_usage_statistics/main.cpp", "memory_pool_resource_usage_statistics.cpp"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_threshold/main.hip", "memory_pool_threshold.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/SOMA/memory_pool_trim/main.cpp", "memory_pool_trim.cpp"),

    # Using-HIP-Runtime-API / Memory-Management / Unified-Memory-Management
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/data_prefetching/main.hip", "data_prefetching.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/dynamic_unified_memory/main.hip", "dynamic_unified_memory.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/explicit_memory/main.hip", "explicit_memory.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/memory_range_attributes/main.hip", "memory_range_attributes.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/standard_unified_memory/main.hip", "standard_unified_memory.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/static_unified_memory/main.hip", "static_unified_memory.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Memory-Management/Unified-Memory-Management/unified_memory_advice/main.hip", "unified_memory_advice.hip"),

    # Using-HIP-Runtime-API / Multi-Device-Management
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_enumeration/main.cpp", "device_enumeration.cpp"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/device_selection/main.hip", "device_selection.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/multi_device_synchronization/main.hip", "multi_device_synchronization.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access/main.hip", "p2p_memory_access.hip"),
    ("HIP-Doc/Programming-Guide/Using-HIP-Runtime-API/Multi-Device-Management/p2p_memory_access_host_staging/main.hip", "p2p_memory_access_host_staging.hip"),

    # Reference / CUDA-to-HIP-API-Function-Comparison
    ("HIP-Doc/Reference/CUDA-to-HIP-API-Function-Comparison/block_reduction/main.cu", "block_reduction.cu"),

    # Reference / HIP-Complex-Math-API
    ("HIP-Doc/Reference/HIP-Complex-Math-API/complex_math/main.hip", "complex_math.hip"),

    # Reference / HIP-Math-API
    ("HIP-Doc/Reference/HIP-Math-API/math/main.hip", "math.hip"),

    # Reference / Low-Precision-Floating-Point-Types
    ("HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp8/main.hip", "low_precision_float_fp8.hip"),
    ("HIP-Doc/Reference/Low-Precision-Floating-Point-Types/low_precision_float_fp16/main.hip", "low_precision_float_fp16.hip"),

    # Tutorials / graph_api
    ("HIP-Doc/Tutorials/graph_api/src/main_streams.hip", "graph_api_tutorial_main_streams.hip"),
    ("HIP-Doc/Tutorials/graph_api/src/main_graph_capture.hip", "graph_api_tutorial_main_graph_capture.hip"),
    ("HIP-Doc/Tutorials/graph_api/src/main_graph_creation.hip", "graph_api_tutorial_main_graph_creation.hip"),
]


def main() -> int:
    failures = []
    for entry in EXAMPLES:
        # Entries are (src, filename) using the default BASE, or
        # (src, filename, base) to override the branch for that file.
        src, filename = entry[0], entry[1]
        base = entry[2] if len(entry) > 2 else BASE
        url = f"{base}/{src}"
        dest = f"{DEST}/{filename}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"OK   {filename}")
        except urllib.error.HTTPError as err:
            print(f"FAIL {filename}: HTTP {err.code} {url}")
            failures.append((filename, url))
        except urllib.error.URLError as err:
            print(f"FAIL {filename}: {err.reason} {url}")
            failures.append((filename, url))

    print(f"\n{len(EXAMPLES) - len(failures)}/{len(EXAMPLES)} downloaded.")
    if failures:
        print(f"{len(failures)} failed:")
        for filename, url in failures:
            print(f"  - {filename}: {url}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
