---
myst:
  html_meta:
    "description": "HIP book"
    "keywords": "HIP, ROCm, Heterogeneous-computing Interface for Portability, HIP textbook, ROCm textbook"
---

<!-- markdownlint-disable MD036 -->

# Accelerated Computing with HIP

The Heterogeneous-computing Interface for Portability (HIP) is a C++ runtime API
and kernel language designed to enable developers to write portable,
high-performance applications that can run on both AMD and NVIDIA GPUs from a
single source code base.

This textbook introduces the core concepts, APIs, and best practices for
accelerated computing using HIP. It provides hands-on guidance for writing GPU
kernels, managing memory, optimizing performance, and integrating HIP with the
broader AMD ROCm ecosystem of tools and libraries.

**Getting started**

* {doc}`./conceptual/foreword`

* {doc}`./conceptual/introduction`

  * {doc}`./conceptual/introduction/parallel_programming`

  * {doc}`./conceptual/introduction/rocm`

  * {doc}`./conceptual/introduction/hip`

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

**Advanced topics**

* {doc}`./tutorial/hip-performance-optimization`

  * {doc}`./tutorial/hip-performance-optimization/reduction`

* {doc}`./how-to/multi-gpu_programming`

  * {doc}`./how-to/multi-gpu/multi_device`

  * {doc}`./how-to/multi-gpu/rccl`

* {doc}`./how-to/deep_learning_with_rocm`

**ROCm platform**

* {doc}`./install/rocm_install`

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

* **Known issues**: Track and report issues on the [HIP book GitHub repository](https://github.com/ROCm/hipbook/issues)
* **Contributing**: See {doc}`Contributing to ROCm docs <rocm:contribute/contributing>` for contribution guidelines
* **License**: Review licensing information on the [Licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page
