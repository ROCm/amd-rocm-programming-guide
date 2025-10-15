.. meta::
  :description: Multi GPU programming
  :keywords: AMD, ROCm, HIP, CUDA, Multi GPU programming,

.. _hip_book_multi_gpu_programming:

********************************************************************************
Multi GPU programming
********************************************************************************

The Multi GPU Programming section explains how to leverage multiple GPUs within
a single system or across nodes to accelerate compute workloads. It covers
techniques for multi-device management, including device selection,
synchronization, and memory handling, as well as the use of RCCL (Radeon
Collective Communication Library) for efficient inter-GPU communication.

These topics help developers scale their HIP applications to fully utilize the
computational power of multi-GPU and distributed environments.

The section includes:

* :doc:`./multi-gpu/multi_device`
* :doc:`./multi-gpu/rccl`
