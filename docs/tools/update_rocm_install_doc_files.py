import urllib.request

repo = "ROCm/rocm-install-on-linux"
branch = "develop"

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/dgl-install.rst",
#      "docs/install/3rd-party/dgl-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/flashinfer-install.rst",
#      "docs/install/3rd-party/flashinfer-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/jax-install.rst",
#      "docs/install/3rd-party/jax-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/llama-cpp-install.rst",
#      "docs/install/3rd-party/llama-cpp-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/megablocks-install.rst",
#      "docs/install/3rd-party/megablocks-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/pytorch-install.rst",
#      "docs/install/3rd-party/pytorch-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/ray-install.rst",
#      "docs/install/3rd-party/ray-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/stanford-megatron-lm-install.rst",
#      "docs/install/3rd-party/stanford-megatron-lm-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/taichi-install.rst",
#      "docs/install/3rd-party/taichi-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/tensorflow-install.rst",
#      "docs/install/3rd-party/tensorflow-install.rst"
# )

# urllib.request.urlretrieve(
#      f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/3rd-party/verl-install.rst",
#      "docs/install/3rd-party/verl-install.rst"
# )

# ROCm installation

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/quick-start.rst",
     "docs/install/quick-start.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/prerequisites.rst",
     "docs/install/prerequisites.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/rocm-offline-installer.rst",
     "docs/install/rocm-offline-installer.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/rocm-runfile-installer.rst",
     "docs/install/rocm-runfile-installer.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install-index.rst",
     "docs/install/install-methods/multi-version-install-index.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager-index.rst",
     "docs/install/install-methods/package-manager-index.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/post-install.rst",
     "docs/install/post-install.rst"
)

# multi version

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-debian.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-debian.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-ol.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-ol.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-rhel.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-rhel.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-rl.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-rl.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-sles.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-sles.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/multi-version-install/multi-version-install-ubuntu.rst",
     "docs/install/install-methods/multi-version-install/multi-version-install-ubuntu.rst"
)

# package-manager

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-debian.rst",
     "docs/install/install-methods/package-manager/package-manager-debian.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-ol.rst",
     "docs/install/install-methods/package-manager/package-manager-ol.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-rhel.rst",
     "docs/install/install-methods/package-manager/package-manager-rhel.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-rl.rst",
     "docs/install/install-methods/package-manager/package-manager-rl.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-sles.rst",
     "docs/install/install-methods/package-manager/package-manager-sles.rst"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/install/install-methods/package-manager/package-manager-ubuntu.rst",
     "docs/install/install-methods/package-manager/package-manager-ubuntu.rst"
)

# images
urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-1-main-menu.png",
     "docs/data/how-to/rocm-offline-installer-1-main-menu.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-2-create-install-options.png",
     "docs/data/how-to/rocm-offline-installer-2-create-install-options.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-3-rocm-options.png",
     "docs/data/how-to/rocm-offline-installer-3-rocm-options.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-3b-rocm-meta.png",
     "docs/data/how-to/rocm-offline-installer-3b-rocm-meta.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-3b-rocm-usecase.png",
     "docs/data/how-to/rocm-offline-installer-3b-rocm-usecase.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-4-driver-options.png",
     "docs/data/how-to/rocm-offline-installer-4-driver-options.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-4b-driver-advanced-options.png",
     "docs/data/how-to/rocm-offline-installer-4b-driver-advanced-options.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-5-extra-packages.png",
     "docs/data/how-to/rocm-offline-installer-5-extra-packages.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-6-post-install.png",
     "docs/data/how-to/rocm-offline-installer-6-post-install.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-7-configuration-1.png",
     "docs/data/how-to/rocm-offline-installer-7-configuration-1.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-8-configuration-2.png",
     "docs/data/how-to/rocm-offline-installer-8-configuration-2.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-offline-installer-9-configuration-3.png",
     "docs/data/how-to/rocm-offline-installer-9-configuration-3.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-driver-menu-4.png",
     "docs/data/how-to/rocm-runfile-driver-menu-4.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-main-menu-1.png",
     "docs/data/how-to/rocm-runfile-main-menu-1.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-postinstall-menu-5.png",
     "docs/data/how-to/rocm-runfile-postinstall-menu-5.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-preinstall-menu-2.png",
     "docs/data/how-to/rocm-runfile-preinstall-menu-2.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-rocm-menu-3.png",
     "docs/data/how-to/rocm-runfile-rocm-menu-3.png"
)

urllib.request.urlretrieve(
     f"https://raw.githubusercontent.com/{repo}/refs/heads/{branch}/docs/data/how-to/rocm-runfile-rocm-menu-uninstall-3b.png",
     "docs/data/how-to/rocm-runfile-rocm-menu-uninstall-3b.png"
)
