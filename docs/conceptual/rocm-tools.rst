.. meta::
  :description: ROCm tools
  :keywords: AMD, ROCm, HIP, CUDA, ROCm Tools,

.. _hip_book_rocm_tools:

********************************************************************************
ROCm tools
********************************************************************************

The latest ROCm tools documentation is available on their official documentation
page. Here, we provide only a list with brief descriptions. For detailed
guidance, please refer to the tools' how-to pages.

System management
-----------------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`AMD SMI <amdsmi:index>`", "System management interface to control AMD GPU settings, monitor performance, and retrieve device and process information"
  ":doc:`ROCm Data Center Tool <rdc:index>`", "Simplifies administration and addresses key infrastructure challenges in AMD GPUs in cluster and data-center environments"
  ":doc:`rocminfo <rocminfo:index>`", "Reports system information"
  ":doc:`ROCm SMI <rocm_smi_lib:index>`", "C library for Linux that provides a user space interface for applications to monitor and control GPU applications"
  ":doc:`ROCm Validation Suite <rocmvalidationsuite:index>`", "Detects and troubleshoots common problems affecting AMD GPUs running in a high-performance computing environment"

Performance
-----------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`ROCm Bandwidth Test <rocm_bandwidth_test:index>`", "Captures the performance characteristics of buffer copying and kernel read/write operations"
  ":doc:`ROCm Compute Profiler <rocprofiler-compute:index>`", "Kernel-level profiling for machine learning and high performance computing (HPC) workloads"
  ":doc:`ROCm Systems Profiler <rocprofiler-systems:index>`", "Comprehensive profiling and tracing of applications running on the CPU or the CPU and GPU"
  ":doc:`ROCProfiler <rocprofiler:index>`", "Profiling tool for HIP applications"
  ":class:`format-big-table` :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>`", "Toolkit for developing analysis tools for profiling and tracing GPU compute applications. This toolkit is in beta and subject to change"
  ":doc:`ROCTracer <roctracer:index>`", "Intercepts runtime API calls and traces asynchronous activity"

.. note::

  `ROCprof Compute Viewer <https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/>`_ is a tool for visualizing and analyzing GPU thread trace data collected using :doc:`rocprofv3 <rocprofiler-sdk:index>`.
  Note that `ROCprof Compute Viewer <https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/>`_ is in an early access state. Running production workloads is not recommended.

Development
-----------

.. csv-table::
  :header: "Component", "Description"

  ":doc:`HIPIFY <hipify:index>`", "Translates CUDA source code into portable HIP C++"
  ":doc:`ROCm CMake <rocmcmakebuildtools:index>`", "Collection of CMake modules for common build and development tasks"
  ":doc:`ROCdbgapi <rocdbgapi:index>`", "ROCm debugger API library"
  ":doc:`ROCm Debugger (ROCgdb) <rocgdb:index>`", "Source-level debugger for Linux, based on the GNU Debugger (GDB)"
  ":doc:`ROCr Debug Agent <rocr_debug_agent:index>`", "Prints the state of all AMD GPU wavefronts that caused a queue error by sending a SIGQUIT signal to the process while the program is running"
