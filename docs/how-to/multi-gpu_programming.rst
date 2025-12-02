.. meta::
  :description: Multi GPU programming
  :keywords: AMD, ROCm, HIP, CUDA, Multi GPU programming,

.. _hip_book_multi_gpu_programming:

********************************************************************************
Multi-GPU programming
********************************************************************************

The multi-GPU Programming section explains how to leverage multiple GPUs within
a single system or across nodes to accelerate compute workloads. It covers
techniques for multi-device management, including device selection,
synchronization, and memory handling. These topics help developers scale HIP
applications to fully utilize the computational power of multi-GPU and distributed
environments.

.. note::
  The ROCm Communication Collectives Library (RCCL) is a stand-alone library that also provides
  multi-GPU and multi-node collective communication primitives optimized for AMD GPUs. It uses
  PCIe and xGMI high-speed interconnects. For more information, see `RCCL documentation <https://rocm.docs.amd.com/projects/rccl/en/latest/index.html>`__.

.. remote-content::
   :repo: ROCm/rocm-systems
   :path: projects/hip/docs/how-to/hip_runtime_api/multi_device.rst
   :default_branch: develop
   :start_line: 10
   :replace: :doc:`ROCm <rocm:what-is-rocm>`|:doc:`ROCm </rocm>`;;data/|../../data/;;../../tools/|../tools/
   :doc_ignore: ./rocm
   :tag_prefix: docs/
   :project_name: HIP
   :docs_base_url: https://rocm.docs.amd.com/projects
