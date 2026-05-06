Uninstalling
============

.. ========================================================== PACKAGE MANAGER ==

.. selected:: i=pkgman

   1. Use your package manager to remove the :ref:`installed packages <rocm-install-rocm>`.

      .. selected:: os=ubuntu os=debian

         .. selected:: gfx=gfx950

            .. code-block:: bash

               sudo apt autoremove amdrocm7.12-gfx950

         .. selected:: gfx=gfx942

            .. code-block:: bash

               sudo apt autoremove amdrocm7.12-gfx94x

         .. selected:: gfx=gfx90a

            .. code-block:: bash

               sudo apt autoremove amdrocm7.12-gfx90a

         .. selected:: gfx=gfx908

            .. code-block:: bash

               sudo apt autoremove amdrocm7.12-gfx908

         .. selected:: gfx=gfx1201 gfx=gfx1200

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx120x amdgpu-lib

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx120x

         .. selected:: gfx=gfx1100 gfx=gfx1101 gfx=gfx1102 gfx=gfx1103

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx110x amdgpu-lib

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx110x

         .. selected:: gfx=gfx1151

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx1151 amdgpu-lib

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx1151

         .. selected:: gfx=gfx1150

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx1150 amdgpu-lib

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo apt autoremove amdrocm7.12-gfx1150

      .. selected:: os=rhel os=oracle-linux os=rocky-linux

         .. selected:: gfx=gfx950

            .. code-block:: bash

               sudo dnf remove amdrocm7.12-gfx950

         .. selected:: gfx=gfx942

            .. code-block:: bash

               sudo dnf remove amdrocm7.12-gfx94x

         .. selected:: gfx=gfx90a

            .. code-block:: bash

               sudo dnf remove amdrocm7.12-gfx90a

         .. selected:: gfx=gfx908

            .. code-block:: bash

               sudo dnf remove amdrocm7.12-gfx908

         .. selected:: gfx=gfx1201 gfx=gfx1200

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo dnf remove amdrocm7.12-gfx120x amdgpu-lib

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo dnf remove amdrocm7.12-gfx120x

         .. selected:: gfx=gfx1100 gfx=gfx1101 gfx=gfx1102 gfx=gfx1103

            .. tab-set::

               .. tab-item:: Graphics and mixed compute
                  :sync: graphics

                  .. code-block:: bash

                     sudo dnf remove amdrocm7.12-gfx110x

               .. tab-item:: Headless compute
                  :sync: compute

                  .. code-block:: bash

                     sudo dnf remove amdrocm7.12-gfx110x

      .. selected:: os=sles

         .. code-block:: bash

            sudo zypper remove amdrocm*

   2. Remove ROCm repositories.

      .. selected:: os=ubuntu os=debian

         .. code-block:: bash

            # Remove ROCm repositories
            sudo rm /etc/apt/sources.list.d/rocm.list

            # Clear the cache and clean the system
            sudo rm -rf /var/cache/apt/*
            sudo apt clean all
            sudo apt update

      .. selected:: os=rhel os=oracle-linux os=rocky-linux

         .. code-block:: bash

            # Remove ROCm repositories
            sudo rm /etc/yum.repos.d/rocm.repo*

            # Clear the cache and clean the system
            sudo rm -rf /var/cache/dnf
            sudo dnf clean all

      .. selected:: os=sles

         .. code-block:: bash

            # Remove ROCm repositories
            sudo zypper removerepo "rocm"

            # Clear the cache and clean the system
            sudo zypper clean --all
            sudo zypper refresh

   3. Remove your ROCm environment configuration from your system.

      .. tab-set::

         .. tab-item:: System-wide
            :sync: env-system-setup

            If you opted for a :ref:`system-wide setup
            <rocm-post-install-env>` during the installation
            process, remove the ROCm environment variables.

            .. code-block:: bash

               sudo rm -f /etc/profile.d/set-rocm-env.sh

         .. tab-item:: User
            :sync: env-user-setup

            If you opted for a :ref:`user-specific setup
            <rocm-post-install-env>` during the installation
            process, remove the ROCm environment configuration block from
            your shell configuration file (``~/.bashrc`` or ``~/.profile``).

.. ====================================================================== PIP ==

.. selected:: i=pip

   1. Clear the pip cache.

      .. selected:: os=ubuntu os=debian os=rhel os=oracle-linux os=rocky-linux os=sles

         .. code-block:: bash

            sudo rm -rf ~/.cache/pip

      .. selected:: os=windows

         .. code-block:: bat

            pip cache purge

   2. Remove your local Python virtual environment.

      .. selected:: os=ubuntu os=debian os=rhel os=oracle-linux os=rocky-linux os=sles

         .. code-block:: bash

            sudo rm -rf .venv

      .. selected:: os=windows

         .. code-block:: bat

            rmdir /s /q .venv

   3. Remove your ROCm environment configuration from your system.

      .. tab-set::

         .. tab-item:: System-wide
            :sync: env-system-setup

            If you opted for a :ref:`system-wide setup
            <rocm-post-install-env>` during the installation
            process, remove the ROCm environment variables.

            .. code-block:: bash

               sudo rm -f /etc/profile.d/set-rocm-env.sh

         .. tab-item:: User
            :sync: env-user-setup

            If you opted for a :ref:`user-specific setup
            <rocm-post-install-env>` during the installation
            process, remove the ROCm environment configuration block from
            your shell configuration file (``~/.bashrc`` or ``~/.profile``).

.. ================================================================== TARBALL ==

.. selected:: i=tar

   .. selected:: os=ubuntu os=debian os=rhel os=oracle-linux os=rocky-linux os=sles

      1. To uninstall ROCm, remove your installation directory.

         .. important::

            The following command assumes you’re working with the
            ``therock-tarball`` directory. If you chose a different directory
            name when :ref:`installing ROCm <rocm-install>`, adjust the command
            accordingly.

         .. code-block:: bash

            sudo rm -rf therock-tarball

      2. Remove your ROCm environment configuration from your system.

         .. tab-set::

            .. tab-item:: System-wide
               :sync: env-system-setup

               If you opted for a :ref:`system-wide setup
               <rocm-post-install-env>` during the installation
               process, remove the ROCm environment variables.

               .. code-block:: bash

                  sudo rm -f /etc/profile.d/set-rocm-env.sh

            .. tab-item:: User
               :sync: env-user-setup

               If you opted for a :ref:`user-specific setup
               <rocm-post-install-env>` during the installation
               process, remove the ROCm environment configuration block from
               your shell configuration file (``~/.bashrc`` or ``~/.profile``).

   .. selected:: os=windows

      1. To uninstall ROCm, remove your installation directory.

         .. code-block:: bat

            rmdir /s /q C:\TheRock

         .. important::

            This step assumes you’re working with the ``C:\TheRock\build``
            directory. If you chose a different directory name when
            :ref:`installing ROCm <rocm-install>`, adjust this step
            accordingly.

      2. **Run command prompt as an administrator** and delete the following environment variables.

         .. code-block:: bat

            setx HIP_DEVICE_LIB_PATH "" /M
            setx HIP_PATH "" /M
            setx HIP_PLATFORM "" /M
            setx LLVM_PATH "" /M

         Remove the following paths from your PATH environment variable using your system settings GUI.
         Navigate to the following screen:

         * Control Panel > System and Security > Edit environment variables

         Edit the PATH variable and delete the following paths:

         * ``C:\TheRock\build\bin``

         * ``C:\TheRock\build\lib\llvm\bin``

      3. To uninstall the Adrenalin Driver, see `Uninstall AMD Software
         <https://www.amd.com/en/resources/support-articles/faqs/RSX2-UNINSTALL.html>`__.


.. ================================================================== RUNFILE ==

.. selected:: i=runfile

   1. Use the following command to uninstall ROCm.

      .. code-block:: bash

         bash rocm-installer-7.12.0-2.run uninstall-rocm

   2. Use the following command to uninstall the AMD GPU Driver (amdgpu).

      .. code-block:: bash

         bash rocm-installer-7.12.0-2.run uninstall-amdgpu
