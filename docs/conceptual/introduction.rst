.. meta::
  :description: HIP book introduction
  :keywords: AMD, ROCm, HIP, CUDA, HIP Book introduction,

.. _hip_book_introduction:

********************************************************************************
Introduction
********************************************************************************

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
