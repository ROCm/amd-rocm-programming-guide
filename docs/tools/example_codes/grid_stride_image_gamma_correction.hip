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
    int global_size = blockDim.x * gridDim.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (; idx < num_values; idx += global_size)
    {
        float value = d_image[idx] / 255.f;
        value = powf(value, gamma);
        d_image[idx] = (std::uint8_t)(value * 255.f);
    }
}

int main(int argc, char* argv[])
{
    int width, height, channels;
    std::uint8_t* data;

    // Load an image from file.

    int blockSize = 256;
    int gridSize = [GridSize];

    float gamma = 4.f;
    image_gamma<<<gridSize, blockSize>>>(d_image, gamma, num_values);

    hipMemcpy(
        data, d_image, num_values * sizeof(std::uint8_t), hipMemcpyDeviceToHost
    );

    // Save the image to disk.

    return EXIT_SUCCESS;
}
