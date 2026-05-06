.. selected:: i=pkgman

   .. selected:: os=ubuntu

      The package manager method uses Ubuntu's native package manager ``apt``
      to install ROCm. This method suits standard system-wide installations
      where ROCm packages should be tracked, updated, and removed through
      system package workflows.

   .. selected:: os=debian

      The package manager method uses Debian's native package manager ``apt``
      to install ROCm. This method suits standard system-wide installations
      where ROCm packages should be tracked, updated, and removed through
      system package workflows.

   .. selected:: os=rhel

      The package manager method uses RHEL's native package manager ``dnf`` to
      install ROCm. This method suits standard system-wide installations where
      ROCm packages should be tracked, updated, and removed through system
      package workflows.

   .. selected:: os=oracle-linux

      The package manager method uses Oracle Linux's native package manager
      ``dnf`` to install ROCm. This method suits standard system-wide
      installations where ROCm packages should be tracked, updated, and removed
      through system package workflows.

   .. selected:: os=rocky-linux

      The package manager method uses Rocky Linux's native package manager
      ``dnf`` to install ROCm. This method suits standard system-wide
      installations where ROCm packages should be tracked, updated, and removed
      through system package workflows.

   .. selected:: os=sles

      The package manager method uses SLES's native package manager ``zypper``
      to install ROCm. This method suits standard system-wide installations
      where ROCm packages should be tracked, updated, and removed through
      system package workflows.

.. selected:: i=pip

   The pip installation method provides ROCm components as Python wheel
   packages in a virtual environment. This method suits Python-focused
   development workflows that use an isolated, per-project ROCm environment
   managed with standard Python packaging tools.

.. selected:: i=tar

   The tarball method provides ROCm as a self-contained installation from
   a pre-built archive. This method suits controlled or restricted environments
   requiring manual placement, updates, and removal outside the system package
   manager.

.. selected:: i=runfile

   The ROCm Runfile Installer can install ROCm and/or the AMD GPU Driver (amdgpu)
   without using a native Linux package management system, making it ideal for
   systems with policy constraints or restricted environments. Network access is
   not needed for install as long as dependencies for ROCm and/or AMD GPU driver
   (amdgpu) are met. A single installer supports all GFX architectures, automates
   post-installation configuration, and offers an interactive command line TUI for
   guided setup.
