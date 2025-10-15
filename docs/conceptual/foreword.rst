.. meta::
  :description: HIP book foreword
  :keywords: AMD, ROCm, HIP, CUDA, HIP Book foreword,

.. _hip_book_foreword:

********************************************************************************
Foreword
********************************************************************************

The world of high-performance computing has recently witnessed a milestone:
achieving, for the first time, exascale performance with the Frontier
supercomputer deployed in the Oak Ridge National Laboratory. Frontier was
superseded as the fastest supercomputer in the world by the El Capitan
supercomputer deployed in the Livermore National Laboratory. These two world’s
fastest supercomputers are powered by AMD’s CPUs, GPUs, and APUs.
Given these advances in computational performance, a new class of applications
can now be pursued, including:

- Weather and climate forecasting,
- Biomedical research,
- High-end equipment development,
- New energy research and exploration,
- Animation design,
- New material research,
- Engineering design, simulation, and analysis,
- Remote sensing data processing, and
- Financial risk analysis.

AMD has enabled these advances by delivering a new class of high-performance
CPUs and GPUs and arichopen-source software stack supporting HIP and ROCm
execution. This emerging programming ecosystem offers many novel features,
including interoperability of hardware accelerators (i.e., AMD and NVIDIA GPUs),
as well support for key high-performance compilers (e.g., LLVM), cluster
deployment, and essential application frameworks (e.g., Raja, Kokkos, TensorFlow
and PyTorch) and key high-performance libraries (rocBLAS, rocSparse, MIOpen,
RCCL,rocFFT). To complement these advances, the high-performance computing
community has also contributed to these milestones by providing state-of-the-art
third-party tools for performance monitoring, debuggers, and visualization tools.
The second edition of Accelerated Computing with HIP, co-authored by Yifan Sun,
Sabila Al Jannat, Trinayan Baruah, and David Kaeli, provides the high
performance computing community with an informative reference to guide
programmers as they leverage the benefits of exascale computing. The text
comprises 13 chapters and three appendices, providing a concise yet complete
reference for HIP programming. Beginning by reviewing the basics of graphics
processors and parallel programming for these devices, it then introduces HIP
kernel programming and the HIP runtime application programming interface (API).
In the following chapters, HIP programming patterns and AMD GPU architectures
are covered. Next, HIP debugging and profiling tools, as well as performance
optimization are presented. This is followed by an examination of ROCm libraries.
Multi-GPU programming, machine learning frameworks, and data-center
computing are discussed subsequently. Finally, several third-party tools are
introduced, and the appendices cover ROCm installation, AMD GPU CDNA assembly
code, and OmniTools. Several HIP and ROCm programming examples are included,
helping the reader quickly master HIP programming.
The textbook is a well-structured introductory approach to accelerated computing
with HIP. Users interested in learning more about scientific computing may refer
to the official documentation website at https://docs.amd.com. While the
textbook helps novice users of HIP acquaint themselves with step-by-step HIP
programming, the ROCm documentation website helps users transition to more
complex APIs, programming models, and development tools. An advanced developer
could find extensive, useful details on the actual implementation of the HIP
programming model and related libraries in the following open-source repository:
https://github.com/ROCm/HIP