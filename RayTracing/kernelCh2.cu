#include <time.h>
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "chapter2.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float* buf, int maxX, int maxY)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= maxX) || (j >= maxY))
    {
        return;
    }

    int pixelIndex = j * maxX * 3 + i * 3;
    buf[pixelIndex + 0] = float(i) / maxX;
    buf[pixelIndex + 1] = float(j) / maxY;
    buf[pixelIndex + 2] = 0.2;
}

void cuImgToFile(const wchar_t* imgFilename, float* buf, Img img)
{
    std::ofstream imgOut;
    imgOut.open(imgFilename);
    imgOut << "P3\n" << img.width << " " << img.height << "\n255\n";

    for (int i = img.height - 1; i >= 0; i--)
    {
        std::cerr << "\Scanlines remaining: " << i << "\n" << std::flush;
        for (int j = 0; j < img.width; j++)
        {
            size_t pixelIndex = i * 3 * img.width + j * 3;

            float r = buf[pixelIndex + 0];
            float g = buf[pixelIndex + 1];
            float b = buf[pixelIndex + 2];

            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);

            imgOut << ir << " " << ig << " " << ib << "\n";
        }
    }

    imgOut.close();
    std::cerr << "Done.\n";
}

int renderGPU(const wchar_t* imgFilename, Img img, Block block)
{
    std::cerr << "Rendering a " << img.width << "x" << img.height << " image ";
    std::cerr << "in " << block.tx << "x" << block.ty << " blocks.\n";

    clock_t start, stop;
    start = clock();

    int nPixels = img.width * img.height;
    size_t bufSize = 3 * nPixels * sizeof(float);

    // allocate FB
    float* buf;
    checkCudaErrors(cudaMallocManaged((void**)&buf, bufSize));

    // Render our buffer
    dim3 blocks(img.width / block.tx + 1, img.height / block.ty + 1);
    dim3 threads(block.tx, block.ty);
    render<<<blocks, threads>>>(buf, img.width, img.height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    stop = clock();
    double timeGPU = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timeGPU << " seconds.\n";

    cuImgToFile(L"cuImage.ppm", buf, img);
    checkCudaErrors(cudaFree(buf));

    return 0;
}