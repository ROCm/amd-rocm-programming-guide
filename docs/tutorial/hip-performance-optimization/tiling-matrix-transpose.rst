.. meta::
  :description: HIP matrix transpose tutorial
  :keywords: AMD, ROCm, HIP, matrix transpose, tiling, coalescing, tutorial

*************************************************************
Tiling and coalescing: matrix transpose
*************************************************************

Matrix transpose is a fundamental linear algebra operation that serves as a
building block for many tasks. At a high level, it does not involve calculations;
it involves only memory movement. Because GPUs typically have significantly
higher memory bandwidth than CPUs, they have a unique advantage when performing
matrix transpose operations.

Like the previous examples, the matrix transpose operation can benefit from a
significantly parallel implementation, in which each thread is responsible for
moving a matrix element.

.. literalinclude:: ../../tools/example_codes/tiling_matrix_transpose.hip
   :language: cuda
   :linenos:
   :start-after: [Sphinx HIP matrix transpose simple kernel start]
   :end-before: [Sphinx HIP matrix transpose simple kernel end]

Memory access patterns
======================

Matrix transpose operations present a unique challenge in terms of memory
access patterns. Theoretically, the number of reads and writes should be
approximately equal, as you write all the data that you read from the input
matrix. However, in practice, the actual memory traffic can be significantly
higher due to the nature of how threads access memory.

To understand this behavior, you need to examine the memory access pattern
carefully. Unlike matrix multiplication where the same data might be accessed
multiple times, in matrix transpose each element is typically accessed only
once. The issue instead lies in the address pattern of the read and write
operations.

In a typical matrix transpose implementation, adjacent threads read data
horizontally (row-wise) and write data vertically (column-wise). Since matrices
are stored in row-major order, adjacent threads read data from contiguous
memory addresses, which allows the GPU to coalesce these read operations into
fewer memory transactions. However, when writing data vertically, the threads
access memory addresses with large strides (the distance between consecutive
elements in a column). This prevents write coalescing, as each thread's write
operation typically requires a separate memory transaction.

As a result, the read operations are coalescable and efficient, while the
write operations are non-coalescable and generate significantly more memory
traffic. This fundamental characteristic of matrix transpose operations makes
them an excellent example for understanding the importance of memory access
patterns and coalescing in GPU programming.

Memory coalescing with LDS
===========================

It is possible to convert non-coalescable memory access to coalescable ones
using the Local Data Store (LDS). The optimization approach splits the matrix
transpose process into two steps. First, you move the data from the input
matrix to LDS memory. For this, you read the data horizontally, ensuring that
the memory access is coalesced. When the LDS buffer is filled, you then move
the data from the LDS to the main memory. In this step, you read the data
vertically and write them horizontally. Since LDS memory has a significantly
higher bandwidth than global memory, reading them horizontally and vertically
does not impact overall performance. However, you write the data horizontally
so that the writes to the main memory can also be coalesced. Thus, you use the
LDS as an intermediate data transfer buffer between the input and output
matrices, allowing you to make both read and write operations coalescable.

The LDS-based kernel implementation uses the same interface as the naive
implementation. The kernel uses a shared memory tile to temporarily store data
before writing it back to the output matrix. This approach ensures that both the
read from global memory and the write to global memory are coalescable, which
significantly improves performance.

.. literalinclude:: ../../tools/example_codes/tiling_matrix_transpose.hip
   :language: cuda
   :linenos:
   :start-after: [Sphinx HIP matrix transpose LDS kernel start]
   :end-before: [Sphinx HIP matrix transpose LDS kernel end]

In the LDS implementation, each threadblock loads a tile of the input matrix
into shared memory in a coalesced fashion. The threads then transpose the
data within the shared memory tile and write it back to the output matrix in a
coalesced manner. This approach effectively solves the non-coalescable write
problem of the naive implementation.

The LDS-based approach provides significant performance benefits because both
read and write operations to global memory are now coalescable, reducing the
total number of memory transactions. By using LDS as an intermediate buffer,
you reduce the pressure on the GPU's memory system and make better use of the
available bandwidth. The tile-based approach improves cache hit rates by
reusing data within the shared memory before writing it back to global memory.

This optimization technique is particularly important for memory-bound kernels
like matrix transpose, where the performance is limited by memory bandwidth
rather than computational throughput.

Implementation guidelines and performance considerations
========================================================

When implementing LDS-based matrix transpose optimization, consider the
following practical aspects:

* **Tile size selection**: Choose tile dimensions that balance LDS memory usage
  and coalescing benefits. Common sizes include 16 × 16 or 32 × 32 tiles, but
  the optimal size depends on your specific GPU architecture

* **Bank conflict avoidance**: Structure your data access patterns to minimize
  memory bank conflicts in the LDS, which can reduce the effectiveness of the
  optimization

* **Memory alignment**: Ensure proper memory alignment for both global memory
  accesses to maximize coalescing benefits

* **Edge case handling**: Account for matrix dimensions that are not perfectly
  divisible by your chosen tile size, implementing proper boundary checks

The performance impact of LDS-based optimization varies depending on matrix
dimensions, GPU architecture, and memory subsystem characteristics. For larger
matrices, the benefits are typically more pronounced as memory bandwidth becomes
the dominant performance factor.
