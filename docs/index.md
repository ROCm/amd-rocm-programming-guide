---
myst:
  html_meta:
    "description": "Start building for HPC and AI with the performance-first AMD ROCm software stack. Explore how-to guides and reference docs."
    "keywords": "Radeon, open, compute, platform, install, how, conceptual, reference, home, docs"
---

# Accelerated Computing with HIP

The Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API
and kernel language designed to enable developers to write portable,
high-performance applications that can run on both AMD and NVIDIA GPUs from a
single source code base.

This textbook introduces the core concepts, APIs, and best practices for
accelerated computing using HIP. It provides hands-on guidance for writing GPU
kernels, managing memory, optimizing performance, and integrating HIP with the
broader AMD ROCm ecosystem of tools and libraries.

* {doc}`./conceptual/foreword`

* {doc}`./conceptual/introduction`

  * {doc}`./conceptual/introduction/parallel_programming`

  * {doc}`./conceptual/introduction/rocm`

  * {doc}`./conceptual/introduction/hip`

* {doc}`./how-to/getting_started_with_hip_programming`

* {doc}`./how-to/hip_runtime_api`

  * {doc}`./how-to/hip_runtime_api/initialization`

  * {doc}`./how-to/hip_runtime_api/memory_management`

  * {doc}`./how-to/hip_runtime_api/error_handling`

  * {doc}`./how-to/hip_runtime_api/asynchronous`

  * {doc}`./how-to/hip_runtime_api/cooperative_groups`

  * {doc}`./how-to/hip_runtime_api/hipgraph`

  * {doc}`./how-to/hip_runtime_api/call_stack`

  * {doc}`./how-to/hip_runtime_api/opengl_interop`

  * {doc}`./how-to/hip_runtime_api/external_interop`

* {doc}`./how-to/performance_optimization`

* {doc}`./how-to/multi-gpu_programming`

  * {doc}`./how-to/multi-gpu/multi_device`

  * {doc}`./how-to/multi-gpu/rccl`

* {doc}`./reference/amd_gpus`

* {doc}`./conceptual/rocm-libraries`

* {doc}`./conceptual/rocm-tools`

* {doc}`./how-to/deep_learning_with_rocm`

* {doc}`./install/rocm_install`

* {doc}`./reference/rocm_in_data_centers`

  * {doc}`./install/containered_rocm`

  * {doc}`./reference/rocm_in_data_centers/rocm_kubernetes`

    * {doc}`./reference/rocm_in_data_centers/kubernetes/configuration`

    * {doc}`./reference/rocm_in_data_centers/kubernetes/examples`

    * {doc}`./reference/rocm_in_data_centers/kubernetes/installation`

    * {doc}`./reference/rocm_in_data_centers/kubernetes/resource-allocation`



Known issues of this material are listed on the [HIP book GitHub repository](https://github.com/ROCm/hipbook/issues).

To contribute to the documentation, refer to {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` page.

You can find licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.
