 .. meta::
  :description: AMD GPUs
  :keywords: AMD, ROCm, HIP, GPUs

.. _hip_book_rocm_gpus:

********************************************************************************
ROCm GPUs
********************************************************************************

This section provides an overview of the AMD GPUs and accelerators supported by
the ROCm software stack. It includes details on the hardware architectures,
their key features, and the corresponding specifications. Note that listed
hardware features indicate architectural capabilities, not performance metrics. 

For performance-related details, refer to the Accelerator and GPU hardware
specifications tables below.

Hardware features
=================

The following table overview of the different hardware architectures and the
features they implement. Hardware features do not imply performance, that
depends on the specifications found in the Accelerator and GPU hardware
specifications page.

.. remote-content::
   :repo: ROCm/hip
   :path: docs/reference/hardware_features.rst
   :start_line: 12
   :default_branch: docs/develop
   :tag_prefix: docs/

Accelerator and GPU hardware specifications
===========================================

.. remote-content::
   :repo: ROCm/ROCm
   :path: docs/reference/gpu-arch-specs.rst
   :start_line: 7
   :default_branch: develop
   :tag_prefix: docs/
