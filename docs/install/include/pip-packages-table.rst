The table below lists the available packages. Each bracketed name is an
optional *extra* of the ``rocm`` meta package — combine the ones you need
as a comma-separated list (for example, ``rocm[libraries,devel]``).

.. matrix::

   .. matrix-row::
      :header:

      .. matrix-cell:: Package

      .. matrix-cell:: Contents

      .. matrix-cell:: Use case

   .. matrix-row::

      .. matrix-cell::

         ``rocm``

      .. matrix-cell:: Core SDK: runtime, HIP, compiler, utility tools, and profiling SDK (rocprofiler-sdk, rocprofv3, roctx).

      .. matrix-cell:: Required by all ROCm users.

   .. matrix-row::

      .. matrix-cell::

         ``rocm[libraries]``

      .. matrix-cell:: Pre-built math and ML host libraries.

      .. matrix-cell:: Required for ML frameworks such as PyTorch and JAX.

   .. matrix-row::

      .. matrix-cell::

         ``rocm[devel]``

      .. matrix-cell:: Compilers, CMake configuration, headers, and static libraries.

      .. matrix-cell:: Building ROCm applications.

   .. matrix-row::

      .. matrix-cell::
         :show-cond: gfx=gfx950

         ``rocm[device-gfx950]``

      .. matrix-cell::
         :show-cond: gfx=gfx942

         ``rocm[device-gfx942]``

      .. matrix-cell::
         :show-cond: gfx=gfx90a

         ``rocm[device-gfx90a]``

      .. matrix-cell::
         :show-cond: gfx=gfx908

         ``rocm[device-gfx908]``

      .. matrix-cell::
         :show-cond: gfx=gfx1201

         ``rocm[device-gfx1201]``

      .. matrix-cell::
         :show-cond: gfx=gfx1200

         ``rocm[device-gfx1200]``

      .. matrix-cell::
         :show-cond: gfx=gfx1100

         ``rocm[device-gfx1100]``

      .. matrix-cell::
         :show-cond: gfx=gfx1101

         ``rocm[device-gfx1101]``

      .. matrix-cell::
         :show-cond: gfx=gfx1102

         ``rocm[device-gfx1102]``

      .. matrix-cell::
         :show-cond: gfx=gfx1103

         ``rocm[device-gfx1103]``

      .. matrix-cell::
         :show-cond: gfx=gfx1030

         ``rocm[device-gfx1030]``

      .. matrix-cell::
         :show-cond: gfx=gfx1151

         ``rocm[device-gfx1151]``

      .. matrix-cell::
         :show-cond: gfx=gfx1150

         ``rocm[device-gfx1150]``

      .. matrix-cell::
         :show-cond: gfx=gfx1152

         ``rocm[device-gfx1152]``

      .. matrix-cell::
         :show-cond: fam=all

         ``rocm[device-all]``

      .. matrix-cell:: Pre-compiled GPU kernels for the specified target.

      .. matrix-cell::

         Required to run GPU workloads; installed alongside ``libraries``.

   .. matrix-row::
      :show-cond: os=ubuntu os=debian os=rhel os=rocky-linux os=oracle-linux os=sles

      .. matrix-cell::

         ``rocm[profiler]``

      .. matrix-cell:: Profiling tools: rocprofiler-systems and rocprofiler-compute.

      .. matrix-cell:: Optional; analyzing and optimizing ROCm applications.
