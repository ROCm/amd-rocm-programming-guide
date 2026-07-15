.. _rocm-install-methods:

.. dropdown:: Compare installation methods
   :animate: fade-in-slide-down
   :color: info
   :icon: tools
   :chevron: down-up

   ROCm offers four installation methods. If you're unsure, start with package
   manager on Linux or tarball on Windows.

   .. list-table::
      :header-rows: 1
      :widths: 30 10 40 30

      * - Install method
        - Platform
        - Best for
        - Install scope
      * - Package manager (apt/dnf/zypper)
        - - Linux
        - - Traditional Linux installation
          - OS managed
          - Auto post-install
        - - System-wide
      * - pip
        - - Linux
          - Windows
        - - Python and ML workflows (PyTorch, JAX)
          - Auto post-install
        - - Python virtual environment
      * - Tarball
        - - Linux
          - Windows
        - - Self-contained, portable setups
          - Monolithic install (all components included)
        - - System-wide
          - Custom install directory
      * - Runfile
        - - Linux
        - - Self-contained, guided (GUI or CLI)
          - Optional offline
          - Packageless
          - Single installer for all GPUs
          - ROCm and amdgpu driver bundled
          - Auto post-install
        - - System-wide
          - Custom install directory
