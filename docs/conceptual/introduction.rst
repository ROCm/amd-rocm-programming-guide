.. meta::
  :description: AMD ROCm Programming Guide introduction
  :keywords: AMD, ROCm, HIP, AMD ROCm Programming Guide introduction,

.. _amd_rocm_programming_guide_introduction:

********************************************************************************
Introduction
********************************************************************************

The rapid growth of high-performance computing (HPC) and artificial intelligence
(AI) has driven an equally rapid evolution in the software ecosystems that power
these workloads. Users increasingly require an open, flexible, and high-performance
GPU compute platform capable of scaling from exploratory prototyping to
mission-critical production workloads. AMD's ROCm open-source software platform
emerged to meet these needs by enabling developers to fully harness the computational
power of AMD Instinct™ GPUs, AMD Radeon™ GPUs, and supported Ryzen™ APUs,
while maintaining compatibility with widely adopted industry frameworks.

Key drivers of ROCm
-----------------------------

Several major trends underpin the adoption and growth of ROCm:

* **Open ecosystem demand**: Organizations—from academic institutions to
  large-scale HPC centers—value transparent, community-driven platforms. ROCm's
  open-source foundation gives developers insight, flexibility, and long-term
  sustainability for their GPU computing stack.

* **HPC and AI convergence**: Modern workloads increasingly blend simulation,
  data analytics, and machine learning. ROCm is designed to accelerate this
  convergence, offering a unified environment that supports scientific
  computing, deep learning, training and inference, large-scale simulations, and
  data-driven modeling.

* **Performance and efficiency**: ROCm is optimized to extract maximum
  performance from AMD GPUs, enabling fine-grained optimizations across 
  on-premise clusters and cloud deployments. 

ROCm users include HPC researchers, AI practitioners, system
integrators, cloud providers expanding GPU compute offerings, and developers
building high-performance workloads on AMD hardware—from workstation users with
desktop Radeon GPUs to large-scale supercomputing installations.

Key features of ROCm tools and the HIP runtime
----------------------------------------------

At the heart of the ROCm ecosystem is the HIP runtime and programming model, a
C++ API introduced by AMD in 2016. HIP provides a familiar, CUDA-like interface
for high-performance development on AMD GPUs. Its design balances performance
with maintainability, allowing developers to write quality parallel code.

Key features of the ROCm tools and HIP runtime include:

* **Rich language interoperability**: ROCm supports multiple programming
  languages and interfaces—including HIP, OpenMP, and OpenCL—offering
  flexibility for different application domains.

* **Optimized math, AI, and HPC libraries**: The platform includes a robust set
  of tuned libraries for BLAS, FFT, sparse linear algebra, random number
  generation, graph analytics, and deep learning primitives.

* **Comprehensive development tools**: ROCm provides profilers, debuggers,
  compilers, and analysis utilities that give developers control over
  performance tuning and hardware-aware optimization.

* **Scalable performance across AMD GPU families**: Applications can seamlessly
  run on AMD Instinct accelerators for datacenters as well as Radeon GPUs for
  workstations and development environments.

* **Integration with industry frameworks**: ROCm supports PyTorch, TensorFlow,
  JAX, and numerous HPC frameworks, allowing developers to adopt GPU
  acceleration without needing to rewrite entire codebases.

Together, HIP and the ROCm toolchain form a cohesive, high-performance
environment for parallel programming, whether targeting deep learning pipelines
or large-scale scientific simulation.

About AMD ROCm programming guide
--------------------------------

This guide serves as a comprehensive programming guide for developers working
with HIP and the broader ROCm ecosystem. It introduces HIP fundamentals,
demonstrates best practices for GPU development, and provides practical
examples that run on any AMD GPU supported by ROCm. While many examples
reference AMD Instinct accelerators such as the MI100, the material remains
portable and applicable across ROCm-supported hardware.

Importantly, this guide is derived from the official `ROCm docs portal <https://rocm.docs.amd.com/en/latest/index.html>`__,
reorganized for PDF export and offline usage. By consolidating and refining the online documentation into a
unified volume, this guide provides developers with a clear, accessible,
and convenient reference.

The chapters that follow introduce the principles of parallel programming on AMD
GPUs, explore the features and ecosystem of HIP, and provide practical guidance
for building efficient, portable, high-performance GPU applications.
