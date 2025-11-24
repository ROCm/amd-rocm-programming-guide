# Configuration options

This document outlines the configuration options available for the AMD GPU device plugin for Kubernetes.

## Environment variables

The device plugin can be configured using the following environment variables:

| Environment Variable | Type | Default | Description |
|-----|------|---------|-------------|
| `AMD_GPU_DEVICE_COUNT` | Integer | Auto-detected | Number of AMD GPUs available on the node |

### Why limit GPU exposure?

There are several reasons you might want to limit the number of GPUs exposed to Kubernetes:

1. **Resource Partitioning**: Reserve some GPUs for non-Kubernetes workloads running on the same node
2. **Testing and Development**: Test applications with restricted GPU access before deploying to production
3. **Mixed Workload Management**: Allocate specific GPUs to different teams or applications based on priority
4. **High Availability**: Keep backup GPUs available for failover scenarios

Setting `AMD_GPU_DEVICE_COUNT` to a value lower than the physical count ensures only a subset of GPUs are made available as Kubernetes resources.

## Command-line flags

The device plugin supports the following command-line flags:

| Flag | Default | Description |
|-----|------|-------------|
| `--kubelet-url` | `http://localhost:10250` | The URL of the Kubelet for device plugin registration |
| `--pulse` | `0` | Time between health check polling in seconds. Set to 0 to disable. |
| `--resource_naming_strategy` | `single` | Resource Naming strategy chosen for k8s resource reporting. |

## Configuration file

You can also provide a configuration file in YAML format to customize the plugin's behavior:

```yaml
gpu:
  device_count: 2
```

### Using the configuration file

To use the configuration file:

1. Create a YAML file with your desired settings (like the example above)
2. Mount this file into the device plugin container

Example deployment snippet:

```yaml
containers:
- image: rocm/k8s-device-plugin
  name: amdgpu-dp-cntr
  env:
  - name: CONFIG_FILE_PATH
    value: "/etc/amdgpu/config.yaml"
  volumeMounts:
  - name: config-volume
    mountPath: /etc/amdgpu
volumes:
- name: config-volume
  configMap:
    name: amdgpu-device-plugin-config
```

With a corresponding `ConfigMap`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: amdgpu-device-plugin-config
  namespace: kube-system
data:
  config.yaml: |
    gpu:
      device_count: 2
```

### Essential volume mounts

The following mounts are required for basic functionality:

| Mount Path | Purpose |
|------------|---------|
| `/var/lib/kubelet/device-plugins` | Required for device plugin registration with the Kubernetes Kubelet |
| `/sys` | Required for GPU detection and topology information |

### Device mounts

For GPU functionality, the following device files must be accessible:

| Mount Path | Purpose |
|------------|---------|
| `/dev/kfd` | Kernel Fusion Driver interface, required for GPU compute workloads |
| `/dev/dri` | Direct Rendering Infrastructure, required for GPU access |

## Example deployments

The repository contains example deployment configurations for different use cases.

### Basic device plugin (k8s-ds-amdgpu-dp.yaml)

A minimal deployment that exposes AMD GPUs to Kubernetes:

- Includes only the essential volume mounts
- Uses minimal security context settings
- Suitable for basic GPU workloads

See the [k8s-ds-amdgpu-dp.yaml deployment file](https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml).

### Enhanced device plugin (k8s-ds-amdgpu-dp-health.yaml)

A more comprehensive deployment of the device plugin that includes additional volume mounts and privileged access for advanced features. This configuration includes:

- Additional volume mounts for `kfd` and `dri` devices
- A dedicated mount for metrics data
- Privileged execution context for direct hardware access

See the [k8s-ds-amdgpu-dp-health.yaml deployment file](https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp-health.yaml).

### Node labeller (k8s-ds-amdgpu-labeller.yaml)

Deploys the AMD GPU node labeller, which adds detailed GPU information as node labels:

- Requires access to `/sys` and `/dev` to gather GPU hardware information
- Creates Kubernetes node labels with details like VRAM size, compute units, etc.
- Helps with GPU-specific workload scheduling

The node labeller can expose labels such as:

- `amd.com/gpu.vram`: GPU memory size
- `amd.com/gpu.cu-count`: Number of compute units
- `amd.com/gpu.device-id`: Device ID of the GPU
- `amd.com/gpu.family`: GPU family/architecture
- `amd.com/gpu.product-name`: Product name of the GPU
- And others based on the passed arguments

Exposing GPU Partition related through Node Labeller:

As part of the arguments passed while starting node labeller, these flags can be passed to expose partition labels:

- compute-partitioning-supported
- memory-partitioning-supported
- compute-memory-partition

These 3 labels have these respective possible values

- `amd.com/compute-partitioning-supported`: `["true", "false"]`
- `amd.com/memory-partitioning-supported`: `["true", "false"]`
- `amd.com/compute-memory-partition`: `["spx_nps1", "cpx_nps1" ,"cpx_nps4", ...]`

See the [k8s-ds-amdgpu-labeller.yaml deployment file](https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-labeller.yaml).

## Resource naming strategy

To customize the way device plugin reports GPU resources to Kubernetes as allocatable k8s resources, use the `single` or `mixed` resource naming strategy flag mentioned above (--resource_naming_strategy)

Before understanding each strategy, please note the definition of homogeneous and heterogeneous nodes

Homogeneous node: A node whose GPUs follow the same compute-memory partition style
    -> Example: A node of 8 GPUs where all 8 GPUs are following CPX-NPS4 partition style

Heterogeneous node: A node whose GPUs follow different compute-memory partition styles
    -> Example: A node of 8 GPUs where 5 GPUs are following SPX-NPS1 and 3 GPUs are following CPX-NPS1

### Single

In `single` mode, the device plugin reports all GPUs (regardless of whether they are whole GPUs or partitions of a GPU) under the resource name `amd.com/gpu`
This mode is supported for homogeneous nodes but not supported for heterogeneous nodes

A node which has 8 GPUs where all GPUs are not partitioned will report its resources as:

```bash
amd.com/gpu: 8
```

A node which has 8 GPUs where all GPUs are partitioned using CPX-NPS4 style will report its resources as:

```bash
amd.com/gpu: 64
```

### Mixed

In `mixed` mode, the device plugin reports all GPUs under a name which matches its partition style.
This mode is supported for both homogeneous nodes and heterogeneous nodes

A node which has 8 GPUs which are all partitioned using CPX-NPS4 style will report its resources as:

```bash
amd.com/cpx_nps4: 64
```

A node which has 8 GPUs where 5 GPUs are following SPX-NPS1 and 3 GPUs are following CPX-NPS1 will report its resources as:

```bash
amd.com/spx_nps1: 5
amd.com/cpx_nps1: 24
```

- If `resource_naming_strategy` is not passed using the flag, then device plugin will internally default to `single` resource naming strategy. This maintains backwards compatibility with earlier release of device plugin with reported resource name of `amd.com/gpu`

- If a node has GPUs which do not support partitioning, such as MI210, then the GPUs are reported under resource name `amd.com/gpu` regardless of the resource naming strategy

Pods can request the resource as per the naming style in their specifications to access AMD GPUs:

```yaml
resources:
  limits:
    amd.com/gpu: 1
```

```yaml
resources:
  limits:
    amd.com/cpx_nps4: 1
```

## Security and access control

### Non-privileged GPU access

For secure workloads, it's recommended to run containers in non-privileged mode while still allowing GPU access. Based on testing with AMD ROCm containers, the following configuration provides reliable non-privileged GPU access:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  hostIPC: true
  containers:
  - name: gpu-container
    image: rocm/pytorch:latest
    resources:
      limits:
        amd.com/gpu: 1
    securityContext:
      # Run as non-privileged container
      privileged: false
      # Prevent privilege escalation
      allowPrivilegeEscalation: false
      # Allow necessary syscalls for GPU operations
      seccompProfile:
        type: Unconfined
```

#### Key security elements

- `privileged: false`: Ensures the container doesn't run with full host privileges
- `allowPrivilegeEscalation: false`: Prevents the process from gaining additional privileges
- `seccompProfile.type: Unconfined`: Allows necessary system calls for GPU operations
