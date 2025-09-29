.. meta::
   :description: How to install deep learning frameworks for ROCm
   :keywords: deep learning, frameworks, ROCm, install, PyTorch, TensorFlow, JAX, MAGMA, DeepSpeed, ML, AI

**********************************
HIP Book
**********************************

Foreword
==================================

The world of high-performance computing has recently witnessed a milestone:
achieving, for the first time, exascale performance with the Frontier
supercomputer deployed in the Oak Ridge National Laboratory. Frontier was
superseded as the fastest supercomputer in the world by the El Capitan
supercomputer deployed in the Livermore National Laboratory. These two world’s
fastest supercomputers are powered by AMD’s CPUs, GPUs, and APUs.
Given these advances in computational performance, a new class of applica tions
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
execution. This emerging programming ecosystem offers many novel features, in
cluding interoperability of hardware accelerators (i.e., AMD and NVIDIA GPUs),
as well support for key high-performance compilers (e.g., LLVM), cluster de
ployment, and essential application frameworks (e.g., Raja, Kokkos, TensorFlow
and PyTorch) and key high-performance libraries (rocBLAS, rocSparse, MIOpen,
RCCL,rocFFT). To complement these advances, the high-performance computing
community has also contributed to these milestones by providing state-of-the-art
third-party tools for performance monitoring, debuggers, and visualization tools.
The second edition of Accelerated Computing with HIP, co-authored by Yifan Sun,
Sabila Al Jannat, Trinayan Baruah, and David Kaeli, provides the high
performance computing community with an informative reference to guide pro
grammers as they leverage the benefits of exascale computing. The text comprises
13 chapters and three appendices, providing a concise yet complete reference for
HIP programming. Beginning by reviewing the basics of graphics processors and
parallel programming for these devices, it then introduces HIP kernel program
ming and the HIP runtime application programming interface (API). In the fol
lowing chapters, HIP programming patterns and AMD GPU architectures are
covered. Next, HIP debugging and profiling tools, as well as performance opti
mization are presented. This is followed by an examination of ROCm libraries.
Multi-GPU programming, machine learning frameworks, and data-center comput
ing are discussed subsequently. Finally, several third-party tools are introduced,
and the appendices cover ROCm installation, AMD GPU CDNA assembly code,
and OmniTools. Several HIP and ROCm programming examples are included,
helping the reader quickly master HIP programming.
The textbook is a well-structured introductory approach to accelerated com
puting with HIP. Users interested in learning more about scientific computing
may refer to the official documentation website at https://docs.amd.com. While
the textbook helps novice users of HIP acquaint themselves with step-by-step HIP
programming, the ROCm documentation website helps users transition to more
complex APIs, programming models, and development tools. An advanced devel
oper could find extensive, useful details on the actual implementation of the HIP
programming model and related libraries in the following open-source repository:
https://github.com/ROCm/HIP

Introduction
==================================

Over the past 40 years, we have seen amazing advances in processing power,
and microprocessor designers have regularly delivered higher performance chips
by adding more transistors and scaling the processor clock, taking advantage of
silicon technology’s Moore’s Law and Dennard scaling. However, early in the 21st
century, as predicted by Robert Dennard at IBM, the clock frequency of a chip was
limited. Hence, we found ourselves unable to push silicon to higher power densities
as the energy accumulated would become impossible to dissipate. In response, chip
vendors began looking for advancements in parallel processing using multiple cores
on a single chip. Although new levels of high performance have been achieved,
most extant software was written assuming a sequential processing model. This
continues to pose challenges to programmers who are pushed to pursue new and
innovative methods to exploit parallelism in their applications.
Recently, we witnessed the number of cores on a single microprocessor grow
from a couple to many. For example, AMD’s third-generation Ryzen Threadripper
central processing unit (CPU) hosts up to 64 cores, with the next iteration aiming
for 128. Application programmers have started to leverage the benefits of many
core CPUs because they excel at running multiple concurrent sequential threads.
Another interesting trend is heterogeneous computing, which uses platforms
specialized for specific execution models. The first wave of such efforts was in
troduced by graphics card vendors (e.g., ATI and NVIDIA) who built the first
graphics processing units (GPUs) with tailored chip designs to accelerate data
parallel graphics-heavy workloads. Notably, these designs required applications
to be written using proprietary graphics languages, which presented barriers to
their widespread use as accelerators.

Today’s graphics vendors typically exploit a single program multiple data (SIMD)
model, in which computational loops are unrolled to leverage parallel execution
units working in a SIMD fashion. With the introduction of programmable
shaders, GPUs could be programmed using high-level languages, leveraging
existing techniques from C and C++, such as NVIDIA’s Compute Unified De
vice Architecture (CUDA) (June 2007) and Khronos’ Open Computing Language (OpenCL)
(August 2009). These parallel programming languages made multi
platform GPU application development fairly consistent. Notably, C++ dialects
use common syntax and data-type conversion standards. Thus, GPU programs now
only differ in their low-level details.

As CUDA gained popularity, concerns were raised about it only running on
NVIDIA hardware, which posed a problematic single-vendor source paradigm.
OpenCL, which can run on GPUs, CPUs, digital signal processors, and field
programmable gate arrays, addressed this issue by adopting a CUDA-like program
ming model. Hence, the cost of portability was significantly reduced. OpenCL’s
requirement that device code being presented as a string posed unnecessary
difficulties with code maintenance and debugging.

For the Fortran 1997 language, Open Multiprocessing (OpenMP) version 4.0
API started supporting GPUs. Currently, it supports the C++03 standard.
However, using anything from C++11 onward can result in unspecified behaviors.
Notably, it forces a portable multithreading procedure, even when directives
dictate automatic data layouts and decompositions, resulting in serious draw
backs. OpenMP also requires the CPU for all processes, as opposed to CUDA and
OpenCL, which outsource parts of the execution (kernels) to the GPU. Further
more, OpenMP only offers the ability to create several threads and change how
blocks of code are executed based on those threads. Moreover, its scalability is
limited by its memory architecture. Experimental results have demonstrated that
OpenMP code performance degrades with large data inputs [42], as opposed to
that of CUDA.

The Open Accelerators (OpenACC) Heterogeneous Programming Standard
appeared in November 2011. As with OpenMP, C, C++, and Fortran source code
can be annotated to identify areas of acceleration using compiler directives and
additional functions. Like OpenMP 4.0 and newer versions, OpenACC targets
both the CPU and GPU for operations. Unfortunately, OpenACC is currently
only supported for PGI and Cray hardware; thus, we cannot fairly compare it to
other heterogeneous technologies.

In August 2012, Microsoft presented its massive parallelism approach as an
extension to the C++ language via its Visual Studio C++ compiler, C++ Ac
celerated Massive Parallelism (AMP). It was implemented on DirectX 11 as an
open specification. A year and a half later, the updated specification (version 1.2)
was released. Microsoft had planned on this update becoming part of the C++14
Standard, but the C++ Committee did not adopt it.

AMD introduced the Heterogeneous Interface for Portability (HIP) program
ming language in October 2016 to address both portability and performance. HIP
follows many similar parallel programming historic conventions that CUDA has
also leveraged. However, HIP can run on multiple platforms with little to no per
formance overhead. Using AMD’s Radeon Open Ecosystem (ROCm) platform,
parallel programs developed using HIP can be used for a wide range of applica
tions, spanning deep learning to molecular dynamics.

This book introduces the HIP programming language and its ecosystem of
libraries and development tools. Notably, it is based on C++, and readers of
this book are expected to be somewhat familiar with the language. In the
examples presented throughout this text, we target the AMD Instinct
Machine-Intelligence (MI)-100 GPU, with which readers are not required to be
familiar. Most code examples will run on any GPU supported by ROCm or CUDA
platforms. This chapter introduces readers to the world of parallel computing
with HIP and ROCm. Later chapters explore the features and ecosystem of HIP

Parallel Programming
-----------------------------------

.. remote-content::
   :repo: ROCm/HIP
   :path: docs/understand/programming_model.rst
   :default_branch: docs/develop
   :tag_prefix: docs/


GPUs
-----------------------------------

ROCm
-----------------------------------

.. include:: ./what-is-rocm.rst

HIP framework
-----------------------------------

What This Book Covers
-----------------------------------

Getting Started with HIP Programming 
====================================

.. remote-content::
   :repo: ROCm/HIP
   :path: docs/tutorial/saxpy.rst
   :start_line: 15
   :default_branch: docs/develop
   :tag_prefix: docs/

HIP Kernel Programming
======================

 3 HIPKernelProgramming 23
 3.1 CallingFunctionswithinHIPKernels . . . . . . . . . . . . . . . . 23
 3.1.1 __global__FunctioninHIP. . . . . . . . . . . . . . . . . 24
 3.1.2 __device__FunctioninHIP. . . . . . . . . . . . . . . . . 24
 3.1.3 __host__FunctioninHIP. . . . . . . . . . . . . . . . . . 25
 3.1.4 Combining__host__and__device__functions . . . . . 26
 3.2 UsingTemplatesinHIPKernels . . . . . . . . . . . . . . . . . . . 27
 3.3 UsingStructsinHIP. . . . . . . . . . . . . . . . . . . . . . . . . . 28
 3.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33


.. remote-content::
   :repo: ROCm/HIP
   :path: docs/how-to/hip_cpp_language_extensions.rst
   :start_line: 15
   :default_branch: docs/develop
   :tag_prefix: docs/




“HelloWorld” inHIP

.. remote-content::
   :repo: ROCm/HIP
   :path: docs/data/env_variables_hip.rst
   :default_branch: docs/develop
   :tag_prefix: docs/


Contents
 1 Introduction 1
 1.1 ParallelProgramming . . . . . . . . . . . . . . . . . . . . . . . . . 3
 1.2 GPUs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
 1.3 ROCm. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
 1.4 HIPFramework. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
 1.5 WhatThisBookCovers . . . . . . . . . . . . . . . . . . . . . . . . 10
 2 GettingStartedwithHIPProgramming 11
  . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

HIP Runtime API usage
=====================

The HIP runtime API provides C and C++ functionalities to manage event, stream,
and memory on GPUs. On the AMD platform, the HIP runtime uses
:doc:`Compute Language Runtime (CLR) <../understand/amd_clr>`, while on NVIDIA
CUDA platform, it is only a thin layer over the CUDA runtime or Driver API.

- **CLR** contains source code for AMD's compute language runtimes: ``HIP`` and
  ``OpenCL™``. CLR includes the ``HIP`` implementation on the AMD
  platform: `hipamd <https://github.com/ROCm/clr/tree/develop/hipamd>`_ and the
  ROCm Compute Language Runtime (``rocclr``). ``rocclr`` is a
  virtual device interface that enables the HIP runtime to interact with
  different backends such as :doc:`ROCr <rocr-runtime:index>` on Linux or PAL on
  Windows. CLR also includes the `OpenCL runtime <https://github.com/ROCm/clr/tree/develop/opencl>`_
  implementation.
- The **CUDA runtime** is built on top of the CUDA driver API, which is a C API
  with lower-level access to NVIDIA GPUs. For details about the CUDA driver and
  runtime API with reference to HIP, see :doc:`CUDA driver API porting guide <../how-to/hip_porting_driver_api>`.

The backends of HIP runtime API under AMD and NVIDIA platform are summarized in
the following figure:

.. figure:: ../data/how-to/hip_runtime_api/runtimes.svg

.. note::

  On NVIDIA platform HIP runtime API calls CUDA runtime or CUDA driver via
  hipother interface. For more information, see the `hipother repository <https://github.com/ROCm/hipother>`_.

Here are the various HIP Runtime API high level functions:

* :doc:`./hip_runtime_api/initialization`
* :doc:`./hip_runtime_api/memory_management`
* :doc:`./hip_runtime_api/error_handling`
* :doc:`./hip_runtime_api/asynchronous`
* :doc:`./hip_runtime_api/cooperative_groups`
* :doc:`./hip_runtime_api/hipgraph`
* :doc:`./hip_runtime_api/call_stack`
* :doc:`./hip_runtime_api/multi_device`
* :doc:`./hip_runtime_api/opengl_interop`
* :doc:`./hip_runtime_api/external_interop`

GPU Programming Patterns
========================

GPUProgrammingPatterns
-----------------------------------

Two-dimensional Kernels
-----------------------------------

Stencils
-----------------------------------

Multi-Kernel Example – BFS
-----------------------------------

CPU-GPU Computing – KMeans
-----------------------------------

Atomic Operations – Histogram
-----------------------------------

Conclusion
-----------------------------------

GPU Internals
========================

 6.1 AMDGPUs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
 6.2 OverallArchitecture . . . . . . . . . . . . . . . . . . . . . . . . . . 75
 6.3 CommandProcessorandtheDMAEngine . . . . . . . . . . . . . 77
 6.4 WorkgroupDispatching . . . . . . . . . . . . . . . . . . . . . . . . 77
 6.5 Sequencer . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
 6.6 SIMDUnit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
 6.7 ThreadDivergence . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
 6.8 MemoryCoalescing. . . . . . . . . . . . . . . . . . . . . . . . . . . 83
 6.9 MemoryHierarchy . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
CONTENTS v
 6.10AMDRDNAGPUs . . . . . . . . . . . . . . . . . . . . . . . . . . 87
 6.11Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
 7 HIPTools 91
 7.1 ROCmInfo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
 7.2 ROCmSMI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
 7.3 TheROCmDebugger . . . . . . . . . . . . . . . . . . . . . . . . . 98
 7.4 ROCmProfiler . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
 7.4.1 ROCTracer . . . . . . . . . . . . . . . . . . . . . . . . . . . 102
 7.4.2 rocprofiler . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
 7.5 ROCmProfilerV2 . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
 7.5.1 ApplicationTracing . . . . . . . . . . . . . . . . . . . . . . 107
 7.5.2 KernelProfiling. . . . . . . . . . . . . . . . . . . . . . . . . 108
 7.5.3 ROCSys . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 111
 7.6 PortingCUDAProgramstoHIPUsingHipify. . . . . . . . . . . . 114
 7.6.1 HipifyTools. . . . . . . . . . . . . . . . . . . . . . . . . . . 114
 7.6.2 GeneralHipifyGuidelines . . . . . . . . . . . . . . . . . . . 119
 7.6.3 HipificationofMatrix-Transpose . . . . . . . . . . . . . . . 120
 7.6.4 CommonPitfallsandSolutions . . . . . . . . . . . . . . . . 123
 7.7 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 124
 8 HIPPerformanceOptimization 125
 8.1 HighlyParallelWorkload–ImageGammaCorrection . . . . . . . 125
 8.2 Fixed-SizedKernels—ImageGammaCorrection. . . . . . . . . . . 128
 8.3 Reduce—ArraySum . . . . . . . . . . . . . . . . . . . . . . . . . . 131
 8.4 Tiling&Reuse–MatrixMultiplication . . . . . . . . . . . . . . . 135
 8.5 Tiling&Coalescing:MatrixTranspose. . . . . . . . . . . . . . . . 140
 8.6 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 143
 9 ROCmLibraries 145
 9.1 rocBLAS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 146
 9.1.1 UsingrocBLAS . . . . . . . . . . . . . . . . . . . . . . . . . 146
 9.1.2 rocBLASfunctions . . . . . . . . . . . . . . . . . . . . . . . 149
 9.1.3 Asynchronousexecution . . . . . . . . . . . . . . . . . . . . 149
 9.1.4 rocBLASonMI100. . . . . . . . . . . . . . . . . . . . . . . 150
 9.1.5 PortingfromthelegacyBLASlibrary . . . . . . . . . . . . 152
 9.2 rocSPARSE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 152
 9.2.1 Sparsedatarepresentation. . . . . . . . . . . . . . . . . . . 153
 9.2.2 rocSPARSEfunctions . . . . . . . . . . . . . . . . . . . . . 154
vi CONTENTS
 9.3 rocFFT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 156
 9.3.1 rocFFTworkflow. . . . . . . . . . . . . . . . . . . . . . . . 157
 9.3.2 FFTExecutionPlan . . . . . . . . . . . . . . . . . . . . . . 159
 9.4 rocRAND . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 160
 9.5 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 164
 10Multi-GPUProgramming 165
 10.1HIPDeviceAPIs . . . . . . . . . . . . . . . . . . . . . . . . . . . . 165
 10.2Stream-BasedMulti-GPUProgramming . . . . . . . . . . . . . . . 167
 10.3Thread-BasedMulti-GPUProgramming . . . . . . . . . . . . . . . 169
 10.4MPI-BasedMulti-GPUProgramming . . . . . . . . . . . . . . . . 171
 10.5GPU–GPUCommunication . . . . . . . . . . . . . . . . . . . . . . 175
 10.6RCCL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 179
 10.6.1 Broadcast . . . . . . . . . . . . . . . . . . . . . . . . . . . . 179
 10.6.2 AllReduce . . . . . . . . . . . . . . . . . . . . . . . . . . . . 182
 10.7Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
 11MachineLearningwithROCm 185
 11.1PyTorchonROCm. . . . . . . . . . . . . . . . . . . . . . . . . . . 186
 11.1.1 InstallingPyTorch . . . . . . . . . . . . . . . . . . . . . . . 186
 11.1.2 TestingthePyTorchInstallation . . . . . . . . . . . . . . . 187
 11.1.3 ImageClassificationusingInceptionV3 . . . . . . . . . . . 188
 11.2TensorFlowonROCm . . . . . . . . . . . . . . . . . . . . . . . . . 190
 11.2.1 InstallingTensorflow. . . . . . . . . . . . . . . . . . . . . . 190
 11.2.2 TestingtheTensorflowInstallation . . . . . . . . . . . . . . 190
 11.2.3 TrainingusingTensorFlow . . . . . . . . . . . . . . . . . . 191
 11.3Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 192
 12ROCminDataCenters 193
 12.1ContainerizedROCm. . . . . . . . . . . . . . . . . . . . . . . . . . 193
 12.2ManagingROCmContainersusingKubernetes . . . . . . . . . . . 194
 12.3ManagingROCmNodesusingSLURM . . . . . . . . . . . . . . . 197
 12.3.1 SLURMinteractivemode . . . . . . . . . . . . . . . . . . . 198
 12.3.2 SLURMbatchsubmissionmode . . . . . . . . . . . . . . . 198
 12.4Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
CONTENTS vii
 13Third-PartyTools 201
 13.1PAPI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 201
 13.1.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . 201
 13.1.2 PAPIutilitiesandtests . . . . . . . . . . . . . . . . . . . . 203
 13.1.3 PAPI supportforAMDGPUs . . . . . . . . . . . . . . . . 204
 13.1.4 PresetEventsandCounterAnalysisToolkit(CAT) . . . . 205
 13.2Score-PandVampir . . . . . . . . . . . . . . . . . . . . . . . . . . 206
 13.2.1 Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . 206
 13.2.2 TracingwithScore-P. . . . . . . . . . . . . . . . . . . . . . 207
 13.2.3 Score-PUsage . . . . . . . . . . . . . . . . . . . . . . . . . 209
 13.2.4 ProfilingtheQuicksilverApplication . . . . . . . . . . . . . 209
 13.2.5 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . 211
 13.3TraceCompassandTheia . . . . . . . . . . . . . . . . . . . . . . . 211
 13.4TAU . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 218
 13.4.1 ProfilingHIPProgramsUsingTAU . . . . . . . . . . . . . 218
 13.4.2 TracingHIPProgramsUsingTAU . . . . . . . . . . . . . . 220
 13.4.3 UsingAPEXtoMeasureHIPPrograms . . . . . . . . . . . 222
 13.4.4 SummaryofTAU . . . . . . . . . . . . . . . . . . . . . . . 224
 13.5TotalViewDebugger . . . . . . . . . . . . . . . . . . . . . . . . . . 225
 13.6HPCToolkit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 230
 13.6.1 HPCToolkit’sWorkflow . . . . . . . . . . . . . . . . . . . . 230
 13.6.2 AnalyzingPIConGPUwithHPCToolkit . . . . . . . . . . . 231
 13.6.3 CollectingandAnalyzingProfilesandTraces . . . . . . . . 231
 13.6.4MeasurementUsingHardwareCounters . . . . . . . . . . . 234
 13.7DebuggingandProfilingwithLinaroForge . . . . . . . . . . . . . 236
 13.7.1 LinaroDDT. . . . . . . . . . . . . . . . . . . . . . . . . . . 237
 13.7.2 LinaroMAP . . . . . . . . . . . . . . . . . . . . . . . . . . 237
 13.7.3 LinaroPerformanceReports. . . . . . . . . . . . . . . . . . 238
 13.7.4 GPUDebuggingUsingLinaroDDT . . . . . . . . . . . . . 238
 13.7.5 GPUProfilingusingLinaroMAP. . . . . . . . . . . . . . . 242
 13.7.6 GPUPerformanceReports . . . . . . . . . . . . . . . . . . 244
 13.8E4S-TheExtremeScaleScientificSoftwareStack . . . . . . . . . 245
 AROCmInstallation 249
 A.1 Prerequisite . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 249
 A.2 UnderstandingtheROCmPackages . . . . . . . . . . . . . . . . . 250
 A.3 Installation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 251
 A.3.1 InstallerScriptMethod . . . . . . . . . . . . . . . . . . . . 251
 A.3.2 PackageManagerMethod . . . . . . . . . . . . . . . . . . . 252
viii CONTENTS
 A.3.3 VerificationoftheInstallationProcess . . . . . . . . . . . . 253
 A.4 UpgradingROCm . . . . . . . . . . . . . . . . . . . . . . . . . . . 254
 A.5 UninstallingROCm . . . . . . . . . . . . . . . . . . . . . . . . . . 254
 BCDNAAssembly 257
 B.1 UsingCDNAAssemblyCode . . . . . . . . . . . . . . . . . . . . . 257
 B.1.1 RetrieveHIPKernelBinary. . . . . . . . . . . . . . . . . . 258
 B.1.2 DisassemblingaCDNABinary . . . . . . . . . . . . . . . . 258
 B.2 CDNARegisters . . . . . . . . . . . . . . . . . . . . . . . . . . . . 258
 B.3 InstructionTypes. . . . . . . . . . . . . . . . . . . . . . . . . . . . 259
 B.4 MemoryAccessInstructions . . . . . . . . . . . . . . . . . . . . . . 260
 B.5 Example: ShiftedCopy . . . . . . . . . . . . . . . . . . . . . . . . 261
 B.6 Example:Branching . . . . . . . . . . . . . . . . . . . . . . . . . . 262
 B.7 ComparingCDNA2andCDNA3 . . . . . . . . . . . . . . . . . . . 264
 B.8 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 266
 COmniTools 267
 C.1 Omnitrace . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 267
 C.1.1 OmnitraceConfigurationFile . . . . . . . . . . . . . . . . . 268
 C.1.2 CollectTraces. . . . . . . . . . . . . . . . . . . . . . . . . . 268
 C.1.3 OutputandVisualization . . . . . . . . . . . . . . . . . . . 270
 C.2 Omniperf . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 270
 C.2.1 ProfilingProgramswithOmniperf . . . . . . . . . . . . . . 271
 C.2.2 AnalysiswithCLI . . . . . . . . . . . . . . . . . . . . . . . 272
 C.2.3 AnalysiswithWeb-BasedGUI . . . . . . . . . . . . . . . . 274
 C.2.4 AnalysiswithGrafana . . . . . . . . . . . . . . . . . . . . . 274
 C.3 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27