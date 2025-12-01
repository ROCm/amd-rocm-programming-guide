// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

__global__ void image_gamma(std::uint8_t* d_image, float gamma, int num_values)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_values)
    {
        d_image[idx] = powf(d_image[idx] / 255.f, gamma) * 255.f;
    }
}

int main()
{
    int width, height, channels, num_values;
    std::uint8_t* data;

    // Load an image from file.

    int blockSize = 256;
    int gridSize = (num_values + blockSize - 1) / blockSize;

    float gamma = 4.f;
    image_gamma<<<gridSize, blockSize>>>(d_image, gamma, num_values);

    hipMemcpy(
        data, d_image, num_values * sizeof(std::uint8_t), hipMemcpyDeviceToHost
    );

    // Save the image to an output file.

    hipFree(d_image);
    return EXIT_SUCCESS;
}
