---
myst:
  html_meta:
    "description": "AMD ROCm programming guide"
    "keywords": "HIP, ROCm, AMD ROCm programming guide, ROCm handbook volume one"
---

<!-- markdownlint-disable MD036 -->

# AMD ROCm Programming Guide

ROCm is a software stack, composed primarily of open-source software, that
provides the tools for programming AMD Graphics Processing Units (GPUs), from
low-level kernels to high-level end-user applications.

The AMD ROCm Programming Guide introduces the core concepts, APIs, and best
practices for programming with ROCm and the HIP programming language. It
provides hands-on guidance for writing GPU kernels, managing memory, optimizing
performance, and integrating HIP with the broader AMD ROCm ecosystem of tools
and libraries.

**Getting started**

* {doc}`./conceptual/introduction`

* {doc}`./conceptual/introduction/rocm`

* {doc}`./conceptual/introduction/hip`

* {doc}`./conceptual/introduction/parallel_programming`

* {doc}`./how-to/getting_started_with_hip_programming`

**Core HIP programming**

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

* {doc}`./tutorial/programming-patterns`

  * {doc}`./tutorial/programming-patterns/matrix_multiplication`

  * {doc}`./tutorial/programming-patterns/atomic_operations_histogram`

  * {doc}`./tutorial/programming-patterns/cpu_gpu_kmeans`

  * {doc}`./tutorial/programming-patterns/stencil_operations`

  * {doc}`./tutorial/programming-patterns/multikernel_bfs`

**Advanced topics**

* {doc}`./tutorial/hip-performance-optimization`

  * {doc}`./tutorial/hip-performance-optimization/reduction`

  * {doc}`./tutorial/hip-performance-optimization/highly-parallel-image-gamma-correction`
  
  * {doc}`./tutorial/hip-performance-optimization/fixed-size-kernels-image-gamma-correction`

* {doc}`./how-to/multi-gpu_programming`

  * {doc}`./how-to/multi-gpu/multi_device`

  * {doc}`./how-to/multi-gpu/rccl`

* {doc}`./how-to/deep_learning_with_rocm`

**ROCm platform**

* {doc}`./install/install`

* {doc}`./reference/amd_gpus`

* {doc}`./conceptual/rocm-libraries`

* {doc}`./conceptual/rocm-tools`

**ROCm in data centers**

* {doc}`./install/containered_rocm`

* {doc}`./reference/rocm_in_data_centers/slurm-integration`

* {doc}`./reference/rocm_in_data_centers/rocm-kubernetes`

  * {doc}`./reference/rocm_in_data_centers/kubernetes/installation`

  * {doc}`./reference/rocm_in_data_centers/kubernetes/configuration`

  * {doc}`./reference/rocm_in_data_centers/kubernetes/resource-allocation`

  * {doc}`./reference/rocm_in_data_centers/kubernetes/examples`

**Additional resources**

* **Known issues**: Track and report issues on the [AMD ROCm programming guide GitHub repository](https://github.com/ROCm/hipbook/issues)
* **Contributing**: See {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` for contribution guidelines
* **License**: Review licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page
