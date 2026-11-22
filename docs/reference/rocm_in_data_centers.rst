.. meta::
  :description: Run ROCm workloads in data centers with containers, the Slurm
                workload manager, and Kubernetes.
  :robots: noindex

.. _hipbook_rocm_in_data_centers:

********************************************************************************
ROCm in data centers
********************************************************************************

Data centers run GPU workloads on clusters of shared nodes. A workload manager
or orchestrator schedules the workloads, and containers keep the software
environment the same on every node. The pages in this section cover the tools
that ROCm supports for these tasks: containers, the Slurm workload manager, and
Kubernetes.

Containers
==========

ROCm Docker containers package ROCm with frameworks and libraries, so every node
runs the same environment. To give a container access to AMD GPUs, use the AMD
Container Runtime Toolkit or pass the GPU device nodes to the container. See
:doc:`Run ROCm Docker containers <rocm:install/docker-containers>`.

Slurm
=====

:doc:`./rocm_in_data_centers/slurm-integration` describes how to configure AMD
Device Metrics Exporter so that it labels GPU metrics with the Slurm job that
uses each GPU. Use the labels to track GPU use per job.

Kubernetes
==========

The :doc:`AMD GPU device plugin for Kubernetes <./rocm_in_data_centers/rocm-kubernetes>`
exposes AMD GPUs to Kubernetes as schedulable resources, so pods can request
GPUs in their resource limits. The optional node labeler adds GPU properties,
such as VRAM size and compute unit count, to node labels for scheduling.

* :doc:`./rocm_in_data_centers/kubernetes/installation`: Install the device
  plugin, with or without health checks, and the node labeler.

* :doc:`./rocm_in_data_centers/kubernetes/configuration`: Configure the device
  plugin with command-line flags and resource naming strategies, and run
  workloads without privileged access.

* :doc:`./rocm_in_data_centers/kubernetes/resource-allocation`: Learn how the
  device plugin selects GPUs and GPU partitions for a request.

* :doc:`./rocm_in_data_centers/kubernetes/examples`: Run example PyTorch and JAX
  workloads on one or more GPUs.

To install and manage the GPU driver, device plugin, node labeler, and Device
Metrics Exporter together, use the
`AMD GPU Operator <https://instinct.docs.amd.com/projects/gpu-operator/en/latest/>`_.
