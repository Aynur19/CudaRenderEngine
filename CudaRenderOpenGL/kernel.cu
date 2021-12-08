#include <Windows.h> // нужно, чтобы библиотеки OpenGL подключались без ошибок

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_gl_interop.h>




class CudaGlBuffer
{
    cudaGraphicsResource* graphRes;
};


int main()
{
    cudaGLSetGLDevice(0);

    return 0;
}

