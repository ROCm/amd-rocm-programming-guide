.. meta::
  :description: HIP book introduction
  :keywords: AMD, ROCm, HIP, CUDA, HIP Book introduction,

.. _hip_book_introduction:

********************************************************************************
Introduction
********************************************************************************


What are the key drivers for ROCm and users of ROCm in the marketplace? 
What are some of the key features of the ROCm tools and HIP runtime? 
What is a good introduction to this document? 

AMD introduced the Heterogeneous Interface for Portability (HIP) programming
language in October 2016 to address both portability and performance. HIP
follows many similar parallel programming historic conventions that CUDA has
also leveraged. However, HIP can run on multiple platforms with little to no
performance overhead. Using AMD’s Radeon Open Ecosystem (ROCm) platform,
parallel programs developed using HIP can be used for a wide range of
applications, spanning deep learning to molecular dynamics.

This book introduces the HIP programming language and its ecosystem of
libraries and development tools. Notably, it is based on C++, and readers of
this book are expected to be somewhat familiar with the language. In the
examples presented throughout this text, we target the AMD Instinct
Machine-Intelligence (MI)-100 GPU, with which readers are not required to be
familiar. Most code examples will run on any GPU supported by ROCm or CUDA
platforms. This chapter introduces readers to the world of parallel computing
with HIP and ROCm. Later chapters explore the features and ecosystem of HIP.
