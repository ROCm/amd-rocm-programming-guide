.. meta::
   :description: Learn how to install AMD ROCm 7.13.0 for supported Instinct GPUs and Ryzen AI APUs on Ubuntu, RHEL, and Windows. This step-by-step guide covers prerequisites, driver setup, installation methods (pip and tarball), and troubleshooting.
   :keywords: AMD ROCm 7.13.0, install ROCm, Instinct GPU, Ryzen APU, Ubuntu, RHEL, Windows, pip install ROCm, ROCm wheel, ROCm tarball, ROCm GPU driver, ROCm setup, ROCm uninstall, ROCm troubleshooting

*************
ROCm packages
*************

Installing ROCm using your Linux distribution's package manager gives you more
granular package installation options. You can tailor your ROCm installation to
your requirements using one or more combinations of ROCm packages and meta
packages. See :doc:`/install/rocm` for detailed installation instructions.

.. =========================================================== GPU/APU FAMILY ==

.. selector:: AMD device family
   :key: fam

   .. selector-option:: Instinct
      :value: instinct
      :width: 3
      :toc-label: AMD Instinct

   .. selector-option:: Radeon PRO
      :value: radeon-pro
      :width: 3
      :toc-label: AMD Radeon PRO

   .. selector-option:: Radeon
      :value: radeon
      :width: 3
      :toc-label: AMD Radeon

   .. selector-option:: Ryzen
      :value: ryzen
      :width: 3
      :toc-label: AMD Ryzen


.. ================================================================ GPU / APU ==

.. selector:: Instinct GPU
   :key: gpu
   :show-cond: fam=instinct

   .. selector-info:: https://www.amd.com/en/products/accelerators/instinct.html

   .. selector-option:: MI355X
      :width: 3
      :toc-label: AMD Instinct MI355X

   .. selector-option:: MI350X
      :width: 3
      :toc-label: AMD Instinct MI350X

   .. selector-option:: MI325X
      :width: 3
      :toc-label: AMD Instinct MI325X

   .. selector-option:: MI300X
      :width: 3
      :toc-label: AMD Instinct MI300X

   .. selector-option:: MI300A
      :width: 20%
      :toc-label: AMD Instinct MI300A

   .. selector-option:: MI250X
      :width: 20%
      :toc-label: AMD Instinct MI250X

   .. selector-option:: MI250
      :width: 20%
      :toc-label: AMD Instinct MI250

   .. selector-option:: MI210
      :width: 20%
      :toc-label: AMD Instinct MI210

   .. selector-option:: MI100
      :width: 20%
      :toc-label: AMD Instinct MI100


.. selector:: Radeon PRO GPU
   :key: gpu
   :show-cond: fam=radeon-pro

   .. selector-info:: https://www.amd.com/en/products/graphics/workstations.html

   .. selector-option:: AI PRO R9700
      :value: ai-r9700
      :width: 3
      :toc-label: AMD Radeon AI PRO R9700

   .. selector-option:: AI PRO R9600D
      :value: ai-r9600d
      :width: 3
      :toc-label: AMD Radeon AI PRO R9600D

   .. selector-option:: W7900 Dual Slot
      :value: w7900-dual-slot
      :width: 3
      :toc-label: AMD Radeon PRO W7900 Dual Slot

   .. selector-option:: W7900
      :value: w7900
      :width: 3
      :toc-label: AMD Radeon PRO W7900

   .. selector-option:: W7800 48GB
      :value: w7800-48gb
      :width: 3
      :toc-label: AMD Radeon PRO W7800 48GB

   .. selector-option:: W7800
      :value: w7800
      :width: 3
      :toc-label: AMD Radeon PRO W7800

   .. selector-option:: W7700
      :value: w7700
      :width: 3
      :toc-label: AMD Radeon PRO W7700

   .. selector-option:: V710
      :value: v710
      :width: 3
      :toc-label: AMD Radeon PRO V710

.. selector:: Radeon GPU
   :key: gpu
   :show-cond: fam=radeon

   .. selector-info:: https://www.amd.com/en/products/graphics/desktops/radeon.html

   .. selector-option:: RX 9070 XT
      :value: rx-9070-xt
      :width: 3
      :toc-label: AMD Radeon RX 9070 XT

   .. selector-option:: RX 9070 GRE
      :value: rx-9070-gre
      :width: 3
      :toc-label: AMD Radeon RX 9070 GRE

   .. selector-option:: RX 9070
      :value: rx-9070
      :width: 3
      :toc-label: AMD Radeon RX 9070

   .. selector-option:: RX 9060 XT LP
      :value: rx-9060-xt-lp
      :width: 3
      :toc-label: AMD Radeon RX 9060 XT LP

   .. selector-option:: RX 9060 XT
      :value: rx-9060-xt
      :width: 3
      :toc-label: AMD Radeon RX 9060 XT

   .. selector-option:: RX 9060
      :value: rx-9060
      :width: 3
      :toc-label: AMD Radeon RX 9060

   .. selector-option:: RX 7900 XTX
      :value: rx-7900-xtx
      :width: 3
      :toc-label: AMD Radeon RX 7900 XTX

   .. selector-option:: RX 7900 XT
      :value: rx-7900-xt
      :width: 3
      :toc-label: AMD Radeon RX 7900 XT

   .. selector-option:: RX 7900 GRE
      :value: rx-7900-gre
      :width: 3
      :toc-label: AMD Radeon RX 7900 GRE

   .. selector-option:: RX 7800 XT
      :value: rx-7800-xt
      :width: 3
      :toc-label: AMD Radeon RX 7800 XT

   .. selector-option:: RX 7700 XT
      :value: rx-7700-xt
      :width: 3
      :toc-label: AMD Radeon RX 7700 XT

   .. selector-option:: RX 7700 XE
      :value: rx-7700-xe
      :width: 3
      :toc-label: AMD Radeon RX 7700 XE

   .. selector-option:: RX 7700
      :value: rx-7700
      :width: 3
      :toc-label: AMD Radeon RX 7700

   .. selector-option:: RX 7600
      :value: rx-7600
      :width: 3
      :toc-label: AMD Radeon RX 7600

.. selector:: Ryzen APU
   :key: gpu
   :show-cond: fam=ryzen

   .. selector-info:: https://www.amd.com/en/products/processors/workstations/mobile.html

   .. selector-option:: AI Max+ PRO 395
      :value: max-pro-395
      :width: 3
      :toc-label: AMD Ryzen AI Max+ PRO 395

   .. selector-option:: AI Max PRO 390
      :value: max-pro-390
      :width: 3
      :toc-label: AMD Ryzen AI Max PRO 390

   .. selector-option:: AI Max PRO 385
      :value: max-pro-385
      :width: 3
      :toc-label: AMD Ryzen AI Max PRO 385

   .. selector-option:: AI Max PRO 380
      :value: max-pro-380
      :width: 3
      :toc-label: AMD Ryzen AI Max PRO 380

   .. selector-option:: AI Max+ 395
      :value: max-395
      :width: 3
      :toc-label: AMD Ryzen AI Max+ 395

   .. selector-option:: AI Max 390
      :value: max-390
      :width: 3
      :toc-label: AMD Ryzen AI Max 390

   .. selector-option:: AI Max 385
      :value: max-385
      :width: 3
      :toc-label: AMD Ryzen AI Max 385

   .. selector-option:: AI 9 HX PRO 475
      :value: 9-hx-pro-475
      :width: 3
      :toc-label: AMD Ryzen AI 9 HX PRO 475

   .. selector-option:: AI 9 HX PRO 470
      :value: 9-hx-pro-470
      :width: 3
      :toc-label: AMD Ryzen AI 9 HX PRO 470

   .. selector-option:: AI 9 PRO 465
      :value: 9-pro-465
      :width: 3
      :toc-label: AMD Ryzen AI 9 PRO 465

   .. selector-option:: AI 7 PRO 450
      :value: 7-pro-450
      :width: 3
      :toc-label: AMD Ryzen AI 7 PRO 450

   .. selector-option:: AI 5 PRO 440
      :value: 5-pro-440
      :width: 3
      :toc-label: AMD Ryzen AI 5 PRO 440

   .. selector-option:: AI 5 PRO 435
      :value: 5-pro-435
      :width: 20%
      :toc-label: AMD Ryzen AI 5 PRO 435

   .. selector-option:: AI 9 HX 375
      :value: 9-hx-375
      :width: 20%
      :toc-label: AMD Ryzen AI 9 HX 375

   .. selector-option:: AI 9 HX 370
      :value: 9-hx-370
      :width: 20%
      :toc-label: AMD Ryzen AI 9 HX 370

   .. selector-option:: AI 9 365
      :value: 9-365
      :width: 20%
      :toc-label: AMD Ryzen AI 9 365

   .. selector-option:: 9 270
      :value: 9-270
      :width: 20%
      :toc-label: AMD Ryzen 9 270

   .. selector-option:: 7 260
      :value: 7-260
      :width: 2
      :toc-label: AMD Ryzen 7 260

   .. selector-option:: 7 250
      :value: 7-250
      :width: 2
      :toc-label: AMD Ryzen 7 250

   .. selector-option:: 5 240
      :value: 5-240
      :width: 2
      :toc-label: AMD Ryzen 5 240

   .. selector-option:: 5 230
      :value: 5-230
      :width: 2
      :toc-label: AMD Ryzen 5 230

   .. selector-option:: 5 220
      :value: 5-220
      :width: 2
      :toc-label: AMD Ryzen 5 220

   .. selector-option:: 3 210
      :value: 3-210
      :width: 2
      :toc-label: AMD Ryzen 3 210

.. selected:: fam=instinct

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300x

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: Debian
         :value: debian
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Oracle Linux
         :value: oracle-linux
         :width: 4

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 4

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi300a

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 20%

      .. selector-option:: Debian
         :value: debian
         :width: 20%

      .. selector-option:: RHEL
         :value: rhel
         :width: 20%
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: Rocky Linux
         :value: rocky-linux
         :width: 20%

      .. selector-option:: SLES
         :value: sles
         :width: 20%
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi250x gpu=mi250

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 3

      .. selector-option:: Debian
         :value: debian
         :width: 3

      .. selector-option:: RHEL
         :value: rhel
         :width: 3
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 3
         :toc-label: SUSE Linux Enterprise Server

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=mi210 gpu=mi100

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 4

      .. selector-option:: RHEL
         :value: rhel
         :width: 4
         :toc-label: Red Hat Enterprise Linux

      .. selector-option:: SLES
         :value: sles
         :width: 4
         :toc-label: SUSE Linux Enterprise Server

.. selected:: fam=radeon-pro

   .. selector:: Operating system
      :key: os
      :show-cond: gpu=ai-r9700 gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 6

      .. selector-option:: RHEL
         :value: rhel
         :width: 6
         :toc-label: Red Hat Enterprise Linux

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=v710 gpu=ai-r9600d

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 6

      .. selector-option:: RHEL
         :value: rhel
         :width: 6
         :toc-label: Red Hat Enterprise Linux

.. selected:: fam=radeon

   .. selector:: Linux distribution
      :key: os
      :show-cond: gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060 gpu=rx-7700 gpu=rx-7600

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 6

      .. selector-option:: RHEL
         :value: rhel
         :width: 6
         :toc-label: Red Hat Enterprise Linux

   .. selector:: Operating system
      :key: os
      :show-cond: gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe

      .. selector-option:: Ubuntu
         :value: ubuntu
         :width: 6

      .. selector-option:: RHEL
         :value: rhel
         :width: 6
         :toc-label: Red Hat Enterprise Linux

.. selector:: Operating system
   :key: os
   :show-cond: fam=ryzen

   .. selector-option:: Ubuntu
      :value: ubuntu
      :width: 12

----

ROCm Core SDK meta packages
===========================

.. _rocm-install-meta-packages:

Meta packages group related components and dependencies together, allowing
you to install only what is necessary for your use case. The following table
describes available ROCm meta packages. Most users should use these.

.. matrix::

   .. matrix-row::
      :header:

      .. matrix-cell:: Meta package name

      .. matrix-cell:: Contents

      .. matrix-cell:: Use case

   .. matrix-row::

      .. matrix-cell::
         :show-cond: gpu=mi355x gpu=mi350x

         ``amdrocm7.13-gfx950``

      .. matrix-cell::
         :show-cond: gpu=mi325x gpu=mi300x gpu=mi300a

         ``amdrocm7.13-gfx94x``

      .. matrix-cell::
         :show-cond: gpu=mi250x gpu=mi250 gpu=mi210

         ``amdrocm7.13-gfx90a``

      .. matrix-cell::
         :show-cond: gpu=mi100

         ``amdrocm7.13-gfx908``

      .. matrix-cell::
         :show-cond: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

         ``amdrocm7.13-gfx120x``

      .. matrix-cell::
         :show-cond: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

         ``amdrocm7.13-gfx110x``

      .. matrix-cell::
         :show-cond: gpu=w6800 gpu=v620

         ``amdrocm7.13-gfx103x``

      .. matrix-cell::
         :show-cond: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

         ``amdrocm7.13-gfx1151``

      .. matrix-cell::
         :show-cond: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

         ``amdrocm7.13-gfx1150``

      .. matrix-cell:: Runtimes, libraries, system control and monitoring tools, and other essential components.

      .. matrix-cell::

         Core runtime environment.
         Install this to run ROCm applications.

   .. matrix-row::

      .. matrix-cell::
         :show-cond: gpu=mi355x gpu=mi350x

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx950``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx950``

      .. matrix-cell::
         :show-cond: gpu=mi325x gpu=mi300x gpu=mi300a

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx94x``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx94x``

      .. matrix-cell::
         :show-cond: gpu=mi250x gpu=mi250 gpu=mi210

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx90a``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx90a``

      .. matrix-cell::
         :show-cond: gpu=mi100

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx908``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx908``

      .. matrix-cell::
         :show-cond: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx120x``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx120x``

      .. matrix-cell::
         :show-cond: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx110x``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx110x``

      .. matrix-cell::
         :show-cond: gpu=w6800 gpu=v620

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx103x``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx103x``

      .. matrix-cell::
         :show-cond: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx1151``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx1151``

      .. matrix-cell::
         :show-cond: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

         .. selected:: os=ubuntu os=debian

            ``amdrocm-core-dev7.13-gfx1150``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-core-devel7.13-gfx1150``

      .. matrix-cell::
         :show-cond: gpu=mi355x gpu=mi350x

         ``amdrocm7.13-gfx950`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=mi325x gpu=mi300x gpu=mi300a

         ``amdrocm7.13-gfx94x`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=mi250x gpu=mi250 gpu=mi210

         ``amdrocm7.13-gfx90a`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=mi100

         ``amdrocm7.13-gfx908`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

         ``amdrocm7.13-gfx120x`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

         ``amdrocm7.13-gfx110x`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=w6800 gpu=v620

         ``amdrocm7.13-gfx103x`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

         ``amdrocm7.13-gfx1151`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::
         :show-cond: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

         ``amdrocm7.13-gfx1150`` plus compilers, CMake configurations, static library files, and headers.

      .. matrix-cell::

         Development environment.
         Install this to build ROCm applications.

   .. matrix-row::

      .. matrix-cell::

         ``amdrocm-developer-tools7.13``

      .. matrix-cell:: Profilers, debuggers, and related tools.

      .. matrix-cell:: Install this to profile, debug, and optimize ROCm applications.

   .. matrix-row::

      .. matrix-cell::

         ``amdrocm-opencl7.13``

      .. matrix-cell:: Components needed to run OpenCL.

      .. matrix-cell:: Install this to run OpenCL applications on ROCm.

   .. matrix-row::

      .. matrix-cell::
         :show-cond: gpu=mi355x gpu=mi350x

         ``amdrocm-core-sdk-gfx950``

      .. matrix-cell::
         :show-cond: gpu=mi325x gpu=mi300x gpu=mi300a

         ``amdrocm-core-sdk-gfx94x``

      .. matrix-cell::
         :show-cond: gpu=mi250x gpu=mi250 gpu=mi210

         ``amdrocm-core-sdk-gfx90a``

      .. matrix-cell::
         :show-cond: gpu=mi100

         ``amdrocm-core-sdk-gfx908``

      .. matrix-cell::
         :show-cond: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

         ``amdrocm-core-sdk-gfx120x``

      .. matrix-cell::
         :show-cond: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

         ``amdrocm-core-sdk-gfx110x``

      .. matrix-cell::
         :show-cond: gpu=w6800 gpu=v620

         ``amdrocm-core-sdk-gfx103x``

      .. matrix-cell::
         :show-cond: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

         ``amdrocm-core-sdk-gfx1151``

      .. matrix-cell::
         :show-cond: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

         ``amdrocm-core-sdk-gfx1150``

      .. matrix-cell:: The complete ROCm Core SDK including runtimes, compilers, development tools, and dependencies.

      .. matrix-cell:: Install this if you need everything.

Math and compute libraries
==========================

.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component

         .. matrix-cell:: Base runtime package

         .. matrix-cell:: Base development package

   .. matrix-row::

      .. matrix-cell:: Composable Kernel

      .. matrix-cell::

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-ck7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-ck7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-ck7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-ck7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-ck7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-ck7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-ck7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-ck7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-ck7.13-gfx1150``

      .. matrix-cell:: ???

   .. matrix-row::

      .. matrix-cell:: hipBLAS

      .. matrix-cell::
         :rowspan: 4

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-blas7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-blas7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-blas7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-blas7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-blas7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-blas7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-blas7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-blas7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-blas7.13-gfx1150``

      .. matrix-cell::
         :rowspan: 4

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-blas-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-blas-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: hipBLASLt

   .. matrix-row::

      .. matrix-cell:: hipSPARSELt

   .. matrix-row::

      .. matrix-cell:: rocBLAS

   .. matrix-row::

      .. matrix-cell:: hipFFT

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-fft7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-fft7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-fft7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-fft7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-fft7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-fft7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-fft7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-fft7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-fft7.13-gfx1150``

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-fft-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-fft-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocFFT

   .. matrix-row::

      .. matrix-cell:: hipRAND

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-rand7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-rand7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-rand7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-rand7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-rand7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-rand7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-rand7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-rand7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-rand7.13-gfx1150``

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rand-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rand-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocRAND

   .. matrix-row::

      .. matrix-cell:: hipSOLVER

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-solver7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-solver7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-solver7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-solver7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-solver7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-solver7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-solver7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-solver7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-solver7.13-gfx1150``

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-solver-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-solver-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocSOLVER

   .. matrix-row::

      .. matrix-cell:: hipSPARSE

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-sparse7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-sparse7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-sparse7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-sparse7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-sparse7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-sparse7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-sparse7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-sparse7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-sparse7.13-gfx1150``

      .. matrix-cell::
         :rowspan: 2

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-sparse-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-sparse-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocSPARSE

   .. matrix-row::

      .. matrix-cell:: rocWMMA

      .. matrix-cell::

         ``amdrocm-math-common7.13``

      .. matrix-cell:: ???

   .. matrix-row::

      .. matrix-cell:: hipCUB

      .. matrix-cell:: ???
         :rowspan: 3

      .. matrix-cell::
         :rowspan: 3

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-ccl-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-ccl-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocPRIM

   .. matrix-row::

      .. matrix-cell:: rocThrust

   .. matrix-row::

      .. matrix-cell:: MIOpen

      .. matrix-cell::

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-dnn7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-dnn7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-dnn7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-dnn7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-dnn7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-dnn7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            ``amdrocm-dnn7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-dnn7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-dnn7.13-gfx1150``

      .. matrix-cell::

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx110x``

         .. selected:: gpu=w6800 gpu=v620

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx103x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx103x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-dnn-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-dnn-devel7.13-gfx1150``

Communication Libraries
========================

.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component name

         .. matrix-cell:: Base runtime package

         .. matrix-cell:: Base development package

   .. matrix-row::

      .. matrix-cell:: RCCL

      .. matrix-cell::

         .. selected:: gpu=mi355x gpu=mi350x

            ``amdrocm-rccl7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            ``amdrocm-rccl7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            ``amdrocm-rccl7.13-gfx90a``

         .. selected:: gpu=mi100

            ``amdrocm-rccl7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            ``amdrocm-rccl7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            ``amdrocm-rccl7.13-gfx110x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            ``amdrocm-rccl7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            ``amdrocm-rccl7.13-gfx1150``

      .. matrix-cell::

         .. selected:: gpu=mi355x gpu=mi350x

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx950``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx950``

         .. selected:: gpu=mi325x gpu=mi300x gpu=mi300a

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx94x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx94x``

         .. selected:: gpu=mi250x gpu=mi250 gpu=mi210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx90a``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx90a``

         .. selected:: gpu=mi100

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx908``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx908``

         .. selected:: gpu=ai-r9700 gpu=ai-r9600d gpu=rx-9070-xt gpu=rx-9070-gre gpu=rx-9070 gpu=rx-9060-xt-lp gpu=rx-9060-xt gpu=rx-9060

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx120x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx120x``

         .. selected:: gpu=w7900-dual-slot gpu=w7900 gpu=w7800-48gb gpu=w7800 gpu=w7700 gpu=v710 gpu=rx-7900-xtx gpu=rx-7900-xt gpu=rx-7900-gre gpu=rx-7800-xt gpu=rx-7700-xt gpu=rx-7700-xe gpu=rx-7700 gpu=rx-7600 gpu=9-270 gpu=7-260 gpu=7-250 gpu=5-240 gpu=5-230 gpu=5-220 gpu=3-210

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx110x``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx110x``

         .. selected:: gpu=max-pro-395 gpu=max-pro-390 gpu=max-pro-385 gpu=max-pro-380 gpu=max-395 gpu=max-390 gpu=max-385

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx1151``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx1151``

         .. selected:: gpu=9-hx-pro-475 gpu=9-hx-pro-470 gpu=9-pro-465 gpu=7-pro-450 gpu=5-pro-440 gpu=5-pro-435 gpu=9-hx-375 gpu=9-hx-370 gpu=9-365

            .. selected:: os=ubuntu os=debian

               ``amdrocm-rccl-dev7.13-gfx1150``

            .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

               ``amdrocm-rccl-devel7.13-gfx1150``

   .. matrix-row::

      .. matrix-cell:: rocSHMEM

      .. matrix-cell:: No package yet

      .. matrix-cell:: No package yet

Runtimes and compilers
======================

.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component name

         .. matrix-cell:: Base runtime package

         .. matrix-cell:: Base development package

   .. matrix-row::

      .. matrix-cell:: HIP

      .. matrix-cell::
         :rowspan: 2

         ``amdrocm-runtime7.13``

      .. matrix-cell::
         :rowspan: 2

         .. selected:: os=ubuntu os=debian

            ``amdrocm-runtime-dev7.13``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-runtime-devel7.13``

   .. matrix-row::

      .. matrix-cell:: ROCr Runtime

   .. matrix-row::

      .. matrix-cell:: HIPIFY

      .. matrix-cell::

         ``amdrocm-hipify7.13``

      .. matrix-cell:: —

   .. matrix-row::

      .. matrix-cell:: LLVM

      .. matrix-cell::

         ``amdrocm-llvm7.13``

      .. matrix-cell::

         .. selected:: os=ubuntu os=debian

            ``amdrocm-llvm-dev7.13``

         .. selected:: os=rhel os=oracle-linux os=rocky-linux os=sles

            ``amdrocm-llvm-devel7.13``

   .. matrix-row::

      .. matrix-cell:: SPIRV-LLVM-Translator

      .. matrix-cell:: ???

      .. matrix-cell:: ???

Profiling and debugging tools
=============================

.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component name

         .. matrix-cell:: Package names

   .. matrix-row::

      .. matrix-cell:: ROCm Compute Profiler (rocprofiler-compute)

      .. matrix-cell::
         :rowspan: 2

         ``amdrocm-profiler7.13``

   .. matrix-row::

      .. matrix-cell:: ROCm Systems Profiler (rocprofiler-systems)

   .. matrix-row::

      .. matrix-cell:: ROCprofiler-SDK

      .. matrix-cell::

         ``amdrocm-profiler-base7.13``

   .. matrix-row::

      .. matrix-cell:: ROCdbgapi

      .. matrix-cell::
         :rowspan: 3

         ``amdrocm-debugger7.13``

   .. matrix-row::

      .. matrix-cell:: ROCm Debugger (ROCgdb)

   .. matrix-row::

      .. matrix-cell:: ROCr Debug Agent

Control and monitoring tools
============================

.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component name

         .. matrix-cell:: Package names

   .. matrix-row::

      .. matrix-cell:: AMD SMI

      .. matrix-cell::

         ``amdrocm-amdsmi7.13``

   .. matrix-row::

      .. matrix-cell:: hipinfo

      .. matrix-cell::
         :rowspan: 2

         ``amdrocm-base7.13``

   .. matrix-row::

      .. matrix-cell:: rocminfo

Expansion
=========


.. matrix::

   .. matrix-head::

      .. matrix-row::
         :header:

         .. matrix-cell:: Component name

         .. matrix-cell:: Package names

   .. matrix-row::

      .. matrix-cell:: ROCm Data Center Tool (RDC)

      .. matrix-cell::

         ``amdrocm-rdc7.13``

   .. matrix-row::

      .. matrix-cell:: RBT/TransferBench

      .. matrix-cell:: ???
