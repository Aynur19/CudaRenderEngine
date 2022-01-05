#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "chapter3.h"

__global__ void render_Ch3(color* buf, int maxX, int maxY)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= maxX) || (j >= maxY))
    {
        return;
    }

    int pixelIndex = j * maxX + i;
    buf[pixelIndex] = color(float(i) / maxX, float(j) / maxY, 0.2f);
}

int renderGPU_Ch3(Img img, Block block)
{
    std::cerr << "Rendering a " << img.width << "x" << img.height << " image ";
    std::cerr << "in " << block.tx << "x" << block.ty << " blocks.\n";

    clock_t start, stop;
    start = clock();

    int nPixels = img.width * img.height;
    size_t bufSize = 3 * nPixels * sizeof(float);

    // allocate FB
    color* buf;
    cudaMallocManaged((void**)&buf, bufSize);

    // Render our buffer
    dim3 blocks(img.width / block.tx + 1, img.height / block.ty + 1);
    dim3 threads(block.tx, block.ty);
    render_Ch3<<<blocks, threads>>>(buf, img.width, img.height);
    cudaDeviceSynchronize();

    stop = clock();
    double timeGPU = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timeGPU << " seconds.\n";

    std::ofstream imgOut;
    imgOut.open(img.cuImgFilename);
    imgOut << "P3\n" << img.width << ' ' << img.height << "\n255\n";

    for (int j = img.height - 1; j >= 0; j--)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < img.width; i++)
        {
            size_t pixelIndex = j * img.width + i;
            writeColor(imgOut, buf[pixelIndex]);
        }
    }
    imgOut.close();
    std::cerr << "Done.\n";

    cudaFree(buf);

    return 0;
}