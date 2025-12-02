.. meta::
  :description: HIP Kernel programming
  :keywords: AMD, ROCm, HIP, CUDA, HIP Kernel Programming

.. _core_hip_programming:

********************************************************************************
Introduction to core HIP programming
********************************************************************************

HIP is AMD's GPU programming interface within the ROCm ecosystem. It provides a C++ runtime API
and kernel language for writing GPU applications. HIP is designed to expose GPU hardware capabilities
directly, enabling developers to implement fine-grained parallelism for scientific computing, simulations,
and AI workloads. 

As described in :ref:`hip_parallel_programming`, the HIP API uses the following features for GPU programming: 

* Kernels: Functions executed in parallel across many GPU threads.
* Thread hierarchy: Threads are organized into blocks, and blocks form a grid.
* Memory spaces: HIP supports global, shared, and constant memory, each optimized for different access patterns.
* Runtime API functions: Device management, memory allocation, kernel launches, synchronization, and error handling.

.. note::
    HIP syntax is similar to NVIDIA CUDA to facilitate porting applications from CUDA to HIP. However, its real strength
    lies in enabling direct, efficient programming for AMD GPUs.

These elements are configured and accessed through the HIP runtime API as described in this section. Beyond the runtime API,
there are specific GPU programming patterns that help to deliver parallelism and performance optimization for your GPU applications.
These programming patterns provide:

* Massive concurrency: Patterns map naturally to thousands of GPU threads.
* Memory efficiency: Shared memory and tiling reduce costly global memory accesses.
* Synchronization tools: HIP provides barriers and atomics to coordinate threads.
* Scalability: Patterns generalize across problem sizes, making them ideal for HPC and AI workloads.

The HIP runtime API gives developers direct control over GPU execution while tutorials on GPU programming patterns provide
practical strategies for implementing parallelism. Together, they form a foundation for building scalable, efficient algorithms
in HPC and AI.
