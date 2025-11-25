.. meta::
  :description: HIP book ROCm libraries
  :keywords: AMD, ROCm, HIP, CUDA, HIP Book ROCm Libraries

.. _hip_book_rocm_libraries:

********************************************************************************
ROCm libraries
********************************************************************************

The latest ROCm libraries documentation is available on their official
documentation page. Here, we provide only a list with brief descriptions.
For detailed guidance, please refer to the libraries' how-to pages.

Machine learning & computer vision
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`Composable Kernel <composable_kernel:index>`", "Provides a programming model for writing performance critical kernels for machine learning workloads across multiple architectures"
  ":doc:`MIGraphX <amdmigraphx:index>`", "Graph inference engine that accelerates machine learning model inference"
  ":doc:`MIOpen <miopen:index>`", "An open source deep-learning library"
  ":doc:`MIVisionX <mivisionx:index>`", "Set of comprehensive computer vision and machine learning libraries, utilities, and applications"
  ":doc:`ROCm Performance Primitives (RPP) <rpp:index>`", "Comprehensive high-performance computer vision library for AMD processors with HIP/OpenCL/CPU back-ends"
  ":doc:`rocAL <rocal:index>`", "An augmentation library designed to decode and process images and videos"
  ":doc:`rocDecode <rocdecode:index>`", "High-performance SDK for access to video decoding features on AMD GPUs"
  ":doc:`rocJPEG <rocjpeg:index>`", "Library for decoding JPG images on AMD GPUs"
  ":doc:`rocPyDecode <rocpydecode:index>`", "Provides access to rocDecode APIs in both Python and C/C++ languages"

.. note::

  `rocCV <https://rocm.docs.amd.com/projects/rocCV/en/latest/index.html>`_  is an efficient GPU-accelerated library for image pre- and post-processing. rocCV is in an early access state. Using it on production workloads is not recommended.

Communication
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`RCCL <rccl:index>`", "Standalone library that provides multi-GPU and multi-node collective communication primitives"
  ":class:`format-big-table` :doc:`rocSHMEM <rocshmem:index>`", "An intra-kernel networking library that provides GPU-centric networking through an OpenSHMEM-like interface"

Math
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  "`half <https://github.com/ROCm/half/>`_", "C++ header-only library that provides an IEEE 754 conformant, 16-bit half-precision floating-point type, along with corresponding arithmetic operators, type conversions, and common mathematical functions"
  ":class:`format-big-table` :doc:`hipBLAS <hipblas:index>`", "BLAS-marshaling library that supports :doc:`rocBLAS <rocblas:index>` and cuBLAS backends"
  ":class:`format-big-table` :doc:`hipBLASLt <hipblaslt:index>`", "Provides general matrix-matrix operations with a flexible API and extends functionalities beyond traditional BLAS library"
  ":doc:`hipFFT <hipfft:index>`", "Fast Fourier transforms (FFT)-marshalling library that supports rocFFT or cuFFT backends"
  ":doc:`hipfort <hipfort:index>`", "Fortran interface library for accessing GPU Kernels"
  ":doc:`hipRAND <hiprand:index>`", "Ports CUDA applications that use the cuRAND library into the HIP layer"
  ":class:`format-big-table` :doc:`hipSOLVER <hipsolver:index>`", "An LAPACK-marshalling library that supports :doc:`rocSOLVER <rocsolver:index>` and cuSOLVER backends"
  ":class:`format-big-table` :doc:`hipSPARSE <hipsparse:index>`", "SPARSE-marshalling library that supports :doc:`rocSPARSE <rocsparse:index>` and cuSPARSE backends"
  ":class:`format-big-table` :doc:`hipSPARSELt <hipsparselt:index>`", "SPARSE-marshalling library with multiple supported backends"
  ":class:`format-big-table` :doc:`rocALUTION <rocalution:index>`", "Sparse linear algebra library for exploring fine-grained parallelism on ROCm runtime and toolchains"
  ":doc:`rocBLAS <rocblas:index>`", "BLAS implementation (in the HIP programming language) on the ROCm runtime and toolchains"
  ":doc:`rocFFT <rocfft:index>`", "Software library for computing fast Fourier transforms (FFTs) written in HIP"
  ":class:`format-big-table` :doc:`rocRAND <rocrand:index>`", "Provides functions that generate pseudorandom and quasirandom numbers"
  ":class:`format-big-table` :doc:`rocSOLVER <rocsolver:index>`", "An implementation of LAPACK routines on ROCm software, implemented in the HIP programming language and optimized for AMD's latest discrete GPUs"
  ":class:`format-big-table` :doc:`rocSPARSE <rocsparse:index>`", "Exposes a common interface that provides BLAS for sparse computation implemented on ROCm runtime and toolchains (in the HIP programming language)"
  ":doc:`rocWMMA <rocwmma:index>`", "C++ library for accelerating mixed-precision matrix multiply-accumulate (MMA) operations"
  ":doc:`Tensile <tensile:src/index>`", "Creates benchmark-driven backend libraries for GEMMs, GEMM-like problems, and general N-dimensional tensor contractions"

Primitives
-----------------------------------------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`hipCUB <hipcub:index>`", "Thin header-only wrapper library on top of :doc:`rocPRIM <rocprim:index>` or CUB that allows project porting using the CUB library to the HIP layer"
  ":doc:`hipTensor <hiptensor:index>`", "AMD's C++ library for accelerating tensor primitives based on the composable kernel library"
  ":doc:`rocPRIM <rocprim:index>`", "Header-only library for HIP parallel primitives"
  ":doc:`rocThrust <rocthrust:index>`", "Parallel algorithm library"