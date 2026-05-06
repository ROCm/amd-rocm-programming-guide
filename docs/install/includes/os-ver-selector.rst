.. selected:: os=ubuntu

   .. selector:: Ubuntu version
      :key: ubuntu-ver

      .. selector-option:: 26.04
         :show-cond: fam=instinct fam=radeon fam=all
         :value: 26.04
         :width: 4

      .. selector-option:: 24.04.4
         :show-cond: fam=instinct fam=radeon fam=all
         :value: 24.04
         :width: 4

      .. selector-option:: 22.04.5
         :show-cond: fam=instinct fam=radeon fam=all
         :value: 22.04
         :width: 4

      .. selector-option:: 24.04.3
         :show-cond: fam=ryzen
         :value: 24.04
         :width: 12


.. =========================================================== DEBIAN VERSION ==

.. selected:: os=debian

   .. selector:: Debian version
      :show-cond: gpu=mi355x gpu=mi325x gpu=mi350x gpu=mi300x
      :key: debian-ver

      .. selector-option:: 13
         :width: 6

      .. selector-option:: 12
         :width: 6

   .. selector:: Debian version
      :show-cond: gpu=mi300a gpu=mi250x gpu=mi250
      :key: debian-ver

      .. selector-option:: 12
         :width: 12


.. ============================================================= RHEL VERSION ==

.. selected:: os=rhel

   .. selector:: RHEL version
      :key: rhel-ver
      :show-cond: fam=instinct fam=radeon fam=all

      .. selector-option:: 10.1
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 10.0
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 9.7
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 9.6
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 9.4
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 8.10
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210 gpu=mi100
         :width: 2

      .. selector-option:: 10.1
         :show-cond: gpu=mi325x
         :width: 20%

      .. selector-option:: 10.0
         :show-cond: gpu=mi325x
         :width: 20%

      .. selector-option:: 9.7
         :show-cond: gpu=mi325x
         :width: 20%

      .. selector-option:: 9.6
         :show-cond: gpu=mi325x
         :width: 20%

      .. selector-option:: 9.4
         :show-cond: gpu=mi325x
         :width: 20%

      .. selector-option:: 10.1
         :show-cond: fam=radeon fam=all
         :width: 6

      .. selector-option:: 9.7
         :show-cond: fam=radeon fam=all
         :width: 6


.. ===================================================== ORACLE LINUX VERSION ==

.. selected:: os=oracle-linux

   .. selector:: Oracle Linux version
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x gpu=mi300x
      :key: oracle-linux-ver

      .. selector-option:: 10
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x
         :width: 6
         :value: 10

      .. selector-option:: 9
         :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x
         :width: 6
         :value: 9

      .. selector-option:: 10
         :show-cond: gpu=mi300x
         :width: 4
         :value: 10

      .. selector-option:: 9
         :show-cond: gpu=mi300x
         :width: 4
         :value: 9

      .. selector-option:: 8
         :show-cond: gpu=mi300x
         :width: 4
         :value: 8


.. ====================================================== ROCKY LINUX VERSION ==

.. selected:: os=rocky-linux

   .. selector:: Rocky Linux version
      :show-cond: gpu=mi300x gpu=mi300a
      :key: rocky-linux-ver

      .. selector-option:: 9
         :width: 12
         :value: 9


.. ============================================================= SLES VERSION ==

.. selected:: os=sles

   .. selector:: SLES version
      :show-cond: gpu=mi355x gpu=mi350x gpu=mi325x gpu=mi300x gpu=mi300a gpu=mi250x gpu=mi250 gpu=mi210
      :key: sles-ver

      .. selector-option:: 16.0
         :width: 6
         :value: 16.0

      .. selector-option:: 15.7
         :width: 15.7

   .. selector:: SLES version
      :show-cond: gpu=mi100
      :key: sles-ver

      .. selector-option:: 15.7
         :width: 12


.. ========================================================== WINDOWS VERSION ==

.. selected:: os=windows

   .. selector:: Windows version
      :key: windows-ver

      .. selector-option:: 11 25H2
         :width: 12
