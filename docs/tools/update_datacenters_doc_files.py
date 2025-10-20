import urllib.request

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/device-metrics-exporter/refs/heads/main/docs/integrations/slurm-integration.md",
     "docs/reference/rocm_in_data_centers/slurm-integration.md"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/k8s-device-plugin/refs/heads/master/docs/index.md",
     "docs/reference/rocm_in_data_centers/rocm-kubernetes.md"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/k8s-device-plugin/refs/heads/master/docs/user-guide/configuration.md",
     "docs/reference/rocm_in_data_centers/kubernetes/configuration.md"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/k8s-device-plugin/refs/heads/master/docs/user-guide/examples.md",
     "docs/reference/rocm_in_data_centers/kubernetes/examples.md"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/k8s-device-plugin/refs/heads/master/docs/user-guide/installation.md",
     "docs/reference/rocm_in_data_centers/kubernetes/installation.md"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/ROCm/k8s-device-plugin/refs/heads/master/docs/user-guide/resource-allocation.md",
     "docs/reference/rocm_in_data_centers/kubernetes/resource-allocation.md"
)