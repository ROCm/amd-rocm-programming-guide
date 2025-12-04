.. meta::
  :robots: noindex

.. _hip_book_kernel_programming:

********************************************************************************
HIP kernel programming
********************************************************************************

As described in the :ref:`hip_parallel_programming`, there are two execution contexts in HIP application code: host and device.
These execution contexts are signified by the use of ``__host__`` and ``__global__`` (or ``__device__``) qualifiers in
the code. The following text describes the various qualifiers that can be defined in your HIP code, and the circumstances
for using these qualifiers. Additional details are provided related to the use of built-in constants, HIP vector types, and
built-in device functions in your code. 

.. remote-content::
   :repo: ROCm/rocm-systems
   :path: projects/hip/docs/how-to/hip_cpp_language_extensions.rst
   :start_line: 15
   :default_branch: develop
   :tag_prefix: docs/
   :project_name: HIP
   :docs_base_url: https://rocm.docs.amd.com/projects


.. TODO:
..  3 HIP Kernel Programming 23
..  3.1 Calling Functions within HIP Kernels . . . . . . . . . . . . . . . . 23
..  3.1.1 __global__ Function in HIP. . . . . . . . . . . . . . . . . 24
..  3.1.2 __device__ Function in HIP. . . . . . . . . . . . . . . . . 24
..  3.1.3 __host__ Function in HIP. . . . . . . . . . . . . . . . . . 25
..  3.1.4 Combining __host__ and __device__ functions . . . . . 26
..  3.2 Using Templates in HIP Kernels . . . . . . . . . . . . . . . . . . . 27
..  3.3 Using Structs in HIP. . . . . . . . . . . . . . . . . . . . . . . . . . 28
..  3.4 Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
