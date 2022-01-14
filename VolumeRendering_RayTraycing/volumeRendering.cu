///////////////////////////////////////////////////////////////////////////////////////////
// ������ ���������� 3D ������� � ������������� ����������� ����� � ���������� ��������. //
///////////////////////////////////////////////////////////////////////////////////////////

// ����������� ����������� ���������.
#include <helper_cuda.h>
#include <helper_math.h>
#include <device_launch_parameters.h>

// ���������� ����� ������ ��� ����� ��������� ��������� � ���.
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

cudaArray *devVolumeArray = 0;              // CUDA-������ ������.
cudaArray *devTransferFuncArray;            // CUDA-������ ��������������.

cudaTextureObject_t	texObject;              // ������ 3D �������� ��� ������ �� CUDA
cudaTextureObject_t transferTex;            // ������ ������������ ������� �������� ��� ������ �� CUDA.

typedef struct float3x4                     // ��������� �������� ������� 3�4 ����� � ��������� �������.
{
    float4 m[3];
};

__constant__ float3x4 constInvViewMatrix;   // ����������� ������ ��� �������� ������� ��������� ����.

struct Ray              // ���.
{
    float3 origin;      // ������ ����.
    float3 direction;   // ����������� ����.
};


/// <summary>
/// ������� CUDA-������� ��� ���������� ����������� ������������ ���� � �����.
/// </summary>
/// <param name="r">����������� ���.</param>
/// <param name="boxMin">��������� ������� ����.</param>
/// <param name="boxMax">������� ������� ����.</param>
/// <param name="tNear">���������� �� ���������� ����������� ���� � ��������.</param>
/// <param name="tFar">���������� �� ������ ���������� ����������� ���� � ��������.</param>
/// <returns>���������� 1 ���� ��� ������� ����������� ����, ����� ���������� 0.</returns>
__device__ int intersectBox(Ray r, float3 boxMin, float3 boxMax, float* tNear, float* tFar)
{
    // ���������� ����������� ���� �� ����� 6-� ��������� ����.
    float3 invR = make_float3(1.0f) / r.direction;
    float3 tBotom = invR * (boxMin - r.origin);
    float3 tTop = invR * (boxMax - r.origin);

    // ������������������ �����������, ����� ����� ���������� � ���������� �� ������ ���.
    float3 tMin = fminf(tTop, tBotom);
    float3 tMax = fmaxf(tTop, tBotom);

    // ���������� ���������� ��������� (largestMinT) � �������� ��������� (smallestMaxT) ��������� �����������.
    float largestMinT = fmaxf(fmaxf(tMin.x, tMin.y), fmaxf(tMin.x, tMin.z));
    float smallestMaxT = fminf(fminf(tMax.x, tMax.y), fminf(tMax.x, tMax.z));

    *tNear = largestMinT;
    *tFar = smallestMaxT;

    return smallestMaxT > largestMinT;
}

/// <summary>
/// ������� �������������� ������� �� ������� (��� ��������).
/// </summary>
__device__ float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

/// <summary>
/// ������� �������������� ������� �� �������.
/// </summary>
__device__ float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

/// <summary>
/// �������������� �������� ����� �� float4 � uint.
/// </summary>
/// <param name="rgba">����������������� ����</param>
/// <returns>���������� ���� ������� � ���� ����� uint.</returns>
__device__ uint rgbaFloatToInt(float4 rgba)
{
    // ��������������� �������� �������� ������� � ���������� [0.0, 1.0].
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);

    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

/// <summary>
/// ������� ���� CUDA ��� ���������� �����������.
/// </summary>
/// <param name="devOutput">������ PBO.</param>
/// <param name="imgWidth">����� ���� (�����������).</param>
/// <param name="imgHeight">������ ���� (�����������).</param>
/// <param name="density">��������� ������.</param>
/// <param name="brightness">�������.</param>
/// <param name="transferOffset">�������� ��������������.</param>
/// <param name="transferScale">������� ��������������.</param>
/// <param name="tex">��������.</param>
/// <param name="transferTex">�������� ��������������.</param>
__global__ void devRender(uint *devOutput, uint imgWidth, uint imgHeight, float density, float brightness, float transferOffset, 
                          float transferScale, cudaTextureObject_t tex, cudaTextureObject_t transferTex)
{
    const int maxSteps = 500;                                   // ������������ ���������� �����.
    const float tStep = 0.01f;                                  // ������ ���������� ����. 
    const float opacityThreshold = 0.95f;                       // ����� ��������������.
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);     // ������ ������� ����.
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);        // ������� ������� ������. 

    uint x = blockIdx.x * blockDim.x + threadIdx.x;             // ������ ������ �� X.
    uint y = blockIdx.y * blockDim.y + threadIdx.y;             // ������ ������ �� Y.

    // ���� ����� ������ ������� �� ������� ���� (�����������), �� ����� ��������� ������.
    if ((x >= imgWidth) || (y >= imgHeight))
    {
        return;
    }

    // ���������� �������������� �������� ������� ����������� �� ���� X � Y.
    float u = (x / (float)imgWidth) * 2.0f - 1.0f;              
    float v = (y / (float)imgHeight) * 2.0f - 1.0f;

    // ���������� ���� ������� � ������� �����������.
    Ray eyeRay;                                                                              
    eyeRay.origin = make_float3(mul(constInvViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f))); 
    eyeRay.direction = normalize(make_float3(u, v, -2.0f));
    eyeRay.direction = mul(constInvViewMatrix, eyeRay.direction);

    // ���������� ����������� � �����
    float tNear, tFar;         // ���������� �� ������ ���� �� ����������� (�������, �������).
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tNear, &tFar);

    // ���� ��� �� ������������������ ����, �� ����� ��������� ������.
    if (!hit)
    {
        return;
    }

    // ���� ��������� ����� ��������� ������ 0, 
    // �� �������������� �� �������� 0 (��������������� �� ���������� ������� �����������.
    if (tNear < 0.0f)
    {
        tNear = 0.0f;     // clamp to near plane
    }

    // ���������� �����, ���������� �� ����.
    float4 sum = make_float4(0.0f);
    float t = tNear;
    float3 pos = eyeRay.origin + eyeRay.direction * tNear;
    float3 step = eyeRay.direction * tStep;

    // �������� �� ������������� ���������� �����.
    for (int i = 0; i < maxSteps; i++)
    {
        // ������ �� 3D �������� � �������������� ������� �� ���������� [0, 1]. 
        float sample = tex3D<float>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f);

        // ��������� �����, ����� �������������� ������������ �������.
        float4 col = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
        col.w *= density;

        // ��������������� ��������� �� �����-�����.
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

        
        sum = sum + col * (1.0f - sum.w);   // �������� ��� ���������� ������.
        if (sum.w > opacityThreshold)       // ���� �������������� ���� ������������� �������, �� ����� �� �����.
        {
            break;
        }

        t += tStep;
        if (t > tFar)                       // ���� �� ��������� ���� �������� �������� ��������� �����, �� ����� �� �����.
        {
            break;
        }
        pos += step;                        // ��������� ������� ������� ����������.
    }
    sum *= brightness;

    devOutput[y * imgWidth + x] = rgbaFloatToInt(sum);  // ������ ����������� ����� � PBO.
}

/// <summary>
/// ��������� ������ ����������.
/// </summary>
/// <param name="bLinearFilter">����, ���������� �� ��������� ��� ������ ������ ����������.</param>
extern "C" void setTextureFilterMode(bool bLinearFilter)
{
    if (texObject)                                              // ���� ������� ������ ��������.
    {
        checkCudaErrors(cudaDestroyTextureObject(texObject));   // �������� ������� ������� �������� �� CUDA.
    }

    // �������� ������� �������� CUDA-�������.
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = devVolumeArray;

    // �������� ������� �������� CUDA-��������.
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;

    // ���������� �������� ������������ ���� ����� ���������� �������. 
    texDescr.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void initCuda(void *hVolume, cudaExtent volumeSize)
{
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();        // �������� ������� �������� ������� ������ CUDA
    
    // ��������� ������ ��� CUDA-������� ������ � ��������� �������� ������� ������ � ��� �������.
    checkCudaErrors(cudaMalloc3DArray(&devVolumeArray, &channelDesc, volumeSize));  
    
    cudaMemcpy3DParms copyParams = {0};                     // �������� �������, ������������ ��������� ����������� 3D ������ CUDA.
    // ��������� ��������� �� ����� ������ ���������
    copyParams.srcPtr = make_cudaPitchedPtr(hVolume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = devVolumeArray;                   // ������������ ������ ������ ����������
    copyParams.extent = volumeSize;                         // ������������ �������������� ������� ����� ������
    copyParams.kind = cudaMemcpyHostToDevice;               // ��������� ���� ��������: ����������� � ����� �� ������
   
    checkCudaErrors(cudaMemcpy3D(&copyParams));             // ����������� ������ ����� 3D ���������

    cudaResourceDesc texRes;                        // ���������� ������� �������� CUDA-�������
    memset(&texRes, 0, sizeof(cudaResourceDesc));   // ���������� CUDA-������� ��������� 0
    texRes.resType = cudaResourceTypeArray;         // ��������� ���� �������: ������
    texRes.res.array.array = devVolumeArray;        // ��������� ������� ������ ������ � ������ ��������

    cudaTextureDesc texDescr;                       // ���������� ������� �������� CUDA-��������
    memset(&texDescr, 0, sizeof(cudaTextureDesc));  // ���������� ������� �������� CUDA-�������� ��������� 0
    texDescr.normalizedCoords = true;               // ��������� ����� �� ������ ������ � ���������������� ����������� ������������
    texDescr.filterMode = cudaFilterModeLinear;     // �������� ������ ����������: �������� ������������ 

    // ��������� ������ ��������� ������ ��������� ���������: ��������� �� ������� ������������ � �������
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    // ��������� ������ ������ ��������: ������ �������� ��� ��������������� ����� � ��������� �������
    texDescr.readMode = cudaReadModeNormalizedFloat;

    // �������� ������� CUDA-��������
    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

    // �������� ������� ������������ �������
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();   // �������� ������� �������� ������� ������ CUDA
    cudaArray* devTransferFuncArray;                                        // ���������� CUDA-������� ��� ������ � ������������ ��������

    // ��������� ������ ��� CUDA-������� ������������ ��������
    checkCudaErrors(cudaMallocArray(&devTransferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));

    // ����������� ������ ������� ������������ ������� �� ����� � CUDA-������� ������������ �������  
    checkCudaErrors(cudaMemcpy2DToArray(devTransferFuncArray, 0, 0, transferFunc, 0, sizeof(transferFunc), 1, cudaMemcpyHostToDevice));

    
    memset(&texRes, 0, sizeof(cudaResourceDesc));       // ��������� ������� �������� CUDA-��������
    texRes.resType = cudaResourceTypeArray;             // ��������� ���� ��� ������� �������� CUDA-��������: ��� �������
    texRes.res.array.array = devTransferFuncArray;      // ���������� CUDA-�������� ������ ������
    
    memset(&texDescr, 0, sizeof(cudaTextureDesc));      // ��������� ������� �������� CUDA-��������
    texDescr.normalizedCoords = true;                   // ��������� ����� �� ������ ������ � ���������������� ����������� ������������ 
    texDescr.filterMode = cudaFilterModeLinear;         // �������� ������ ����������: �������� ������������
    texDescr.addressMode[0] = cudaAddressModeClamp;     // ��������� ������ ��������� ������ ��������� ���������: ��������� �� ������� ������������ � �������
    texDescr.readMode = cudaReadModeElementType;        // ��������� ������ ������ ��������: ������ �������� ��� ��������� ��� ��������

    // �������� ������� CUDA-��������
    checkCudaErrors(cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
}

extern "C" void freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(texObject));       // ����������� ������� �������� �� CUDA.
    checkCudaErrors(cudaDestroyTextureObject(transferTex));
    checkCudaErrors(cudaFreeArray(devVolumeArray));             // ����������� ������, ����������� �� CUDA-������.
    checkCudaErrors(cudaFreeArray(devTransferFuncArray));
}

extern "C" void renderKernel(dim3 gridSize, dim3 blockSize, uint *devOutput, uint imgWidth, uint imgHeight,
                             float density, float brightness, float transferOffset, float transferScale)
{
    devRender<<<gridSize, blockSize>>>(devOutput, imgWidth, imgHeight, density, brightness,
                                      transferOffset, transferScale, texObject, transferTex);
}

extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    // ������ ������ � ����� � ����������� ������ CUDA.
    checkCudaErrors(cudaMemcpyToSymbol(constInvViewMatrix, invViewMatrix, sizeofMatrix));
}
