# Resource allocation

## Overview

The [Device Plugin](https://github.com/ROCm/k8s-device-plugin) daemon set discovers and makes AMD GPUs available to the Kubernetes cluster. Allocation logic determines which set of GPUs or resources are allocated when a job or pod requests them. The allocation logic can run an algorithm to determine which GPUs should be picked from the available ones.

### Allocator package

The Device Plugin has an allocator package where you can define multiple policies on how the allocation should be done. Each policy can follow a different algorithm to decide the allocation strategy based on system needs. Actual allocation of AMD GPUs is done by Kubernetes and Kubelet. The allocation policy only decides the GPUs to be picked from the available GPUs for any given request.

### Best-effort allocation policy

The `best-effort` policy is used as the default allocation policy. This policy chooses GPUs based on the topology of the GPUs to ensure optimal affinity and better performance. During the initialization phase, the Device Plugin calculates a score for every pair of GPUs and stores it in memory. This score is calculated based on the following criteria:

- Type of connectivity link between the pair. Most common AMD GPU deployments use either XGMI or PCIe links to connect the GPUs. `XGMI` connectivity offers better performance than PCIe connectivity. The score assigned for a pair connected using XGMI is lower than that of a pair connected using PCIe (lower score is better).
- [NUMA affinity](https://rocm.blogs.amd.com/software-tools-optimization/affinity/part-1/README.html) of the GPU pair. GPU pairs that are part of the same NUMA domain get a lower score than pairs from different NUMA domains.
- For scenarios that involve partitioned GPUs, partitions from the same GPU are assigned a better score than partitions from different GPUs.

When an allocation request for size S comes, the allocator calculates all subsets of size S from the available GPUs. For each set, the score is maintained (based on the above criteria). The set with the lowest score is picked for allocation. At any given time, the best-effort policy tries to provide the best possible combination of GPUs from the available GPU pool.

The following rules are followed for allocation requests for X GPU partitions:

- The allocator tries to allocate all partitions from the same GPU if possible.
- If a GPU with fewer available partitions can accommodate the request, that GPU is preferred. This maximizes the utilization of GPUs already in use for other workloads and helps avoid fragmentation of unused GPUs.
- If more than one GPU is needed to accommodate the request, the allocator considers the topology (link type and NUMA affinity) as described above and generates all possible subsets. The subset with the lowest weight among the possible candidates is allocated.
