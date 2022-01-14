///////////////////////////////////////////////////////////////////////////////////////////
// Пример рендеринга 3D объекта с использование трассировки лучей и трехмерной текстуры. //
///////////////////////////////////////////////////////////////////////////////////////////

// Подулючение необходимых библиотек.
#include <helper_cuda.h>
#include <helper_math.h>
#include <device_launch_parameters.h>

// объявление типов данных для более короткого обращения к ним.
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

cudaArray *devVolumeArray = 0;              // CUDA-массив объема.
cudaArray *devTransferFuncArray;            // CUDA-массив преобразования.

cudaTextureObject_t	texObject;              // Объект 3D текстуры для работы на CUDA
cudaTextureObject_t transferTex;            // Объект передаточной функции текстуры для работы на CUDA.

typedef struct float3x4                     // Структура хранения матрицы 3х4 чисел с плавающей запятой.
{
    float4 m[3];
};

__constant__ float3x4 constInvViewMatrix;   // Константная память для хранения матрицы обратного вида.

struct Ray              // Луч.
{
    float3 origin;      // Начало луча.
    float3 direction;   // Направление луча.
};


/// <summary>
/// Функция CUDA-девайса для нахождения пересечения испускаемого луча с кубом.
/// </summary>
/// <param name="r">Испускаемый луч.</param>
/// <param name="boxMin">Ближайшая граница куба.</param>
/// <param name="boxMax">Дальняя граница куба.</param>
/// <param name="tNear">Расстояние до ближайшего пересечения луча с объектом.</param>
/// <param name="tFar">Расстояние до самого удаленного пересечения луча с объектом.</param>
/// <returns>Возвращает 1 если луч пересек поверхность куба, иначе фозвращает 0.</returns>
__device__ int intersectBox(Ray r, float3 boxMin, float3 boxMax, float* tNear, float* tFar)
{
    // Вычисление пересечений луча со всеми 6-ю сторонами куба.
    float3 invR = make_float3(1.0f) / r.direction;
    float3 tBotom = invR * (boxMin - r.origin);
    float3 tTop = invR * (boxMax - r.origin);

    // Переупорядочивание пересечений, чтобы найти наименьшее и наибольшее на каждой оси.
    float3 tMin = fminf(tTop, tBotom);
    float3 tMax = fmaxf(tTop, tBotom);

    // нахождение наибольлее удаленной (largestMinT) и наименее удаленной (smallestMaxT) координат пересечения.
    float largestMinT = fmaxf(fmaxf(tMin.x, tMin.y), fmaxf(tMin.x, tMin.z));
    float smallestMaxT = fminf(fminf(tMax.x, tMax.y), fminf(tMax.x, tMax.z));

    *tNear = largestMinT;
    *tFar = smallestMaxT;

    return smallestMaxT > largestMinT;
}

/// <summary>
/// Функция преобразования вектора по матрице (без перевода).
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
/// Функция преобразования вектора по матрице.
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
/// Преобразование значений цвета из float4 в uint.
/// </summary>
/// <param name="rgba">Преобразовываемый цвет</param>
/// <returns>Возвращает цвет пикселя в виде числа uint.</returns>
__device__ uint rgbaFloatToInt(float4 rgba)
{
    // Масштабирование значений цветовых каналов в промежуток [0.0, 1.0].
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);

    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

/// <summary>
/// Функция ядра CUDA для рендеринга изображения.
/// </summary>
/// <param name="devOutput">Данные PBO.</param>
/// <param name="imgWidth">Длина окна (изображения).</param>
/// <param name="imgHeight">Ширина окна (изображения).</param>
/// <param name="density">Плотность данных.</param>
/// <param name="brightness">Яркость.</param>
/// <param name="transferOffset">Смещение преобразования.</param>
/// <param name="transferScale">Масштаб преобразования.</param>
/// <param name="tex">Текстура.</param>
/// <param name="transferTex">Текстура преобразования.</param>
__global__ void devRender(uint *devOutput, uint imgWidth, uint imgHeight, float density, float brightness, float transferOffset, 
                          float transferScale, cudaTextureObject_t tex, cudaTextureObject_t transferTex)
{
    const int maxSteps = 500;                                   // Максимальное количество шагов.
    const float tStep = 0.01f;                                  // Размер приращения шага. 
    const float opacityThreshold = 0.95f;                       // Порог непрозрачности.
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);     // Нижней граница куба.
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);        // Верхняя граница порога. 

    uint x = blockIdx.x * blockDim.x + threadIdx.x;             // Индекс потока по X.
    uint y = blockIdx.y * blockDim.y + threadIdx.y;             // Индекс потока по Y.

    // Если номер потока выходин за границы окна (изображения), то поток завершает работу.
    if ((x >= imgWidth) || (y >= imgHeight))
    {
        return;
    }

    // Вычисление номализованных значений вектора направления по осям X и Y.
    float u = (x / (float)imgWidth) * 2.0f - 1.0f;              
    float v = (y / (float)imgHeight) * 2.0f - 1.0f;

    // Вычисление луча взгляда в мировых координатах.
    Ray eyeRay;                                                                              
    eyeRay.origin = make_float3(mul(constInvViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f))); 
    eyeRay.direction = normalize(make_float3(u, v, -2.0f));
    eyeRay.direction = mul(constInvViewMatrix, eyeRay.direction);

    // Нахождение пересечения с полем
    float tNear, tFar;         // Расстояния от начала луча до пересечения (ближнее, дальнее).
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tNear, &tFar);

    // Если луч не пересекповерхность куба, то поток завершает работу.
    if (!hit)
    {
        return;
    }

    // Если ближайшая точка оказалась меньше 0, 
    // то приравнивается ее значению 0 (устанавливается на координату ближней поверхности.
    if (tNear < 0.0f)
    {
        tNear = 0.0f;     // clamp to near plane
    }

    // Накопление цвета, направлясь по лучу.
    float4 sum = make_float4(0.0f);
    float t = tNear;
    float3 pos = eyeRay.origin + eyeRay.direction * tNear;
    float3 step = eyeRay.direction * tStep;

    // Итерация по максимальному количеству шагов.
    for (int i = 0; i < maxSteps; i++)
    {
        // Чтение из 3D текстуры и переназначение позиции на координаты [0, 1]. 
        float sample = tex3D<float>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, pos.z * 0.5f + 0.5f);

        // Получение цвета, путем преобразования передаточной функции.
        float4 col = tex1D<float4>(transferTex, (sample - transferOffset) * transferScale);
        col.w *= density;

        // Предварительное умножение на альфа-канал.
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

        
        sum = sum + col * (1.0f - sum.w);   // Оператор для смешивания цветов.
        if (sum.w > opacityThreshold)       // Если непрозрачность выше установленной границы, то выход из цикла.
        {
            break;
        }

        t += tStep;
        if (t > tFar)                       // Если на следующем шаге пройдена наиболее удаленная точка, то выход из цикла.
        {
            break;
        }
        pos += step;                        // Изменение позиции текущей координаты.
    }
    sum *= brightness;

    devOutput[y * imgWidth + x] = rgbaFloatToInt(sum);  // запись полученного цвета в PBO.
}

/// <summary>
/// Установка режима фильтрации.
/// </summary>
/// <param name="bLinearFilter">Флаг, отвечающий за установку или снятие режима фильтрации.</param>
extern "C" void setTextureFilterMode(bool bLinearFilter)
{
    if (texObject)                                              // Если имеется объект текстуры.
    {
        checkCudaErrors(cudaDestroyTextureObject(texObject));   // Удаление данного объекта текстуры из CUDA.
    }

    // Создание объекта описания CUDA-ресурса.
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = devVolumeArray;

    // Создание объекта описания CUDA-текстуры.
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = true;

    // Применение линейной интерполяции если режим фильтрации включен. 
    texDescr.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
    texDescr.addressMode[0] = cudaAddressModeWrap;
    texDescr.addressMode[1] = cudaAddressModeWrap;
    texDescr.addressMode[2] = cudaAddressModeWrap;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));
}

extern "C" void initCuda(void *hVolume, cudaExtent volumeSize)
{
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();        // создание объекта описания формата канала CUDA
    
    // Выделение памяти для CUDA-массива объема с указанием описания формата канала и его размера.
    checkCudaErrors(cudaMalloc3DArray(&devVolumeArray, &channelDesc, volumeSize));  
    
    cudaMemcpy3DParms copyParams = {0};                     // создание объекта, описывающего параметры копирования 3D памяти CUDA.
    // получение указателя на адрес памяти источника
    copyParams.srcPtr = make_cudaPitchedPtr(hVolume, volumeSize.width * sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = devVolumeArray;                   // присваивание адреса памяти назначения
    copyParams.extent = volumeSize;                         // призваивание запрашиваемого размера копии памяти
    copyParams.kind = cudaMemcpyHostToDevice;               // устанавка типа операции: копирование с хоста на девайс
   
    checkCudaErrors(cudaMemcpy3D(&copyParams));             // копирование данных между 3D объектами

    cudaResourceDesc texRes;                        // объявление объекта описания CUDA-ресурса
    memset(&texRes, 0, sizeof(cudaResourceDesc));   // заполнение CUDA-ресурса значением 0
    texRes.resType = cudaResourceTypeArray;         // установка типа ресурса: массив
    texRes.res.array.array = devVolumeArray;        // установка массива данных объема в массив ресурсов

    cudaTextureDesc texDescr;                       // объявление объекта описания CUDA-текстуры
    memset(&texDescr, 0, sizeof(cudaTextureDesc));  // заполнение объекта описания CUDA-текстуры значением 0
    texDescr.normalizedCoords = true;               // установка флага на чтение данных с нормализованными текстурными координатами
    texDescr.filterMode = cudaFilterModeLinear;     // установк режима фильтрации: линейная интерполяция 

    // установка режима адресации границ координат текустуры: выходящие за границы приравнивать к крайним
    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    // установка режима чтения текстуры: читать текстуру как нормализованное число с плавающей запятой
    texDescr.readMode = cudaReadModeNormalizedFloat;

    // создание объекта CUDA-текстуры
    checkCudaErrors(cudaCreateTextureObject(&texObject, &texRes, &texDescr, NULL));

    // создание массива передаточной функции
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

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();   // создание объекта описания формата канала CUDA
    cudaArray* devTransferFuncArray;                                        // объявление CUDA-массива для работы с передаточной функцией

    // выделение памяти для CUDA-массива передаточной функцией
    checkCudaErrors(cudaMallocArray(&devTransferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));

    // копирование данных массива передаточной функции из хоста в CUDA-массива передаточной функции  
    checkCudaErrors(cudaMemcpy2DToArray(devTransferFuncArray, 0, 0, transferFunc, 0, sizeof(transferFunc), 1, cudaMemcpyHostToDevice));

    
    memset(&texRes, 0, sizeof(cudaResourceDesc));       // обнуление объекта описания CUDA-текстуры
    texRes.resType = cudaResourceTypeArray;             // установки типа для объекта описания CUDA-текстуры: тип массива
    texRes.res.array.array = devTransferFuncArray;      // присвоение CUDA-текстуре массив данных
    
    memset(&texDescr, 0, sizeof(cudaTextureDesc));      // обнуление объекта описания CUDA-текстуры
    texDescr.normalizedCoords = true;                   // установка флага на чтение данных с нормализованными текстурными координатами 
    texDescr.filterMode = cudaFilterModeLinear;         // установк режима фильтрации: линейная интерполяция
    texDescr.addressMode[0] = cudaAddressModeClamp;     // установка режима адресации границ координат текустуры: выходящие за границы приравнивать к крайним
    texDescr.readMode = cudaReadModeElementType;        // установка режима чтения текстуры: читать текстуру как указанный тип элемента

    // создание объекта CUDA-текстуры
    checkCudaErrors(cudaCreateTextureObject(&transferTex, &texRes, &texDescr, NULL));
}

extern "C" void freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(texObject));       // Уничтожение объекта текстуры на CUDA.
    checkCudaErrors(cudaDestroyTextureObject(transferTex));
    checkCudaErrors(cudaFreeArray(devVolumeArray));             // Освобожение памяти, выделенного на CUDA-массив.
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
    // запись данных с хоста в константную память CUDA.
    checkCudaErrors(cudaMemcpyToSymbol(constInvViewMatrix, invViewMatrix, sizeofMatrix));
}
