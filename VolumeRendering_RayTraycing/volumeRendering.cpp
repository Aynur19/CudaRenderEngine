////////////////////////////////////////////////////////////////////////////////////////////
// Пример отображение 3D объекта с использование трассировки лученй и трехмерных текстур. //
////////////////////////////////////////////////////////////////////////////////////////////

// Подключение библиотек OpenGL.
#include <helper_gl.h>
#include <GL/freeglut.h>

// Подключение библиотек CUDA.
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// Подключение вспомогательных утилит.
#include <helper_cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

// Объявление типов данных для более короткого обращения к ним
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned char VolumeType;

const char* volumeFilename = "Heart256.raw";                    // Название считываемого файла.
cudaExtent volumeSize = make_cudaExtent(256, 256, 256);         // Размеры объема для рендеринга.

uint width = 512;           // Длина окна.
uint height = 512;          // Ширина окна.

dim3 blockSize(16, 16);     // Размеры блока.
dim3 gridSize;              // Размеры сетки.

float3 viewRotation;                                    // Вектор поворота.
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);  // Вектор перемещения.

float invViewMatrix[12];        // Матрица транспонированная (инвертированная).
float density = 0.05f;          // Плотность (прозрачность).
float brightness = 1.0f;        // Яркость.
float transferOffset = 0.0f;    // Смещение преобразования (по сути сила проникновения луча).
float transferScale = 1.0f;     // Масштаб преобразования (для масштабирования объекта на сцене).
bool linearFiltering = true;    // Флаг линейной фильтрации.

GLuint pbo = 0;     // Объект OpenGL PBO.  
GLuint tex = 0;     // Объект текстуры OpenGL.   

// Указатель на графические ресурсы CUDA (для преобразования в Pixel Buffer Object (PBO)).
struct cudaGraphicsResource* cudaResourcePBO;
StopWatchInterface* timer = 0;  // Указатель на таймер.

int *pArgc;     // Указатель на количество аргументов командной строки, поданных при запуске программы.
char **pArgv;   // Указатель на аргументы командной строки, поданные при запуске программы.

/// <summary>
/// Установка режима фильтрации.
/// </summary>
/// <param name="bLinearFilter">Флаг, отвечающий за установку или снятие режима фильтрации.</param>
extern "C" void setTextureFilterMode(bool bLinearFilter);

/// <summary>
/// Инициализация CUDA-массивов и их настройка.
/// </summary>
/// <param name="hVolume">Данные.</param>
/// <param name="volumeSize">Размеры объема.</param>
extern "C" void initCuda(void *hVolume, cudaExtent volumeSize);

/// <summary>
/// Освобожение памяти из буферов на CUDA.
/// </summary>
extern "C" void freeCudaBuffers();

/// <summary>
/// Функция вызова ядра CUDA редеринга.
/// </summary>
/// <param name="gridSize">Размер сетки.</param>
/// <param name="blockSize">Размер блока.</param>
/// <param name="devOutput">Данные PBO.</param>
/// <param name="imgWidth">Длина изображения.</param>
/// <param name="imgHeight">Ширина изображения.</param>
/// <param name="density">Плотность даных.</param>
/// <param name="brightness">Яркость.</param>
/// <param name="transferOffset">Смещение преобразования.</param>
/// <param name="transferScale">Масштаб преобразования.</param>
extern "C" void renderKernel(dim3 gridSize, dim3 blockSize, uint * devOutput, uint imageW, uint imageH,
                             float density, float brightness, float transferOffset, float transferScale);

/// <summary>
/// Функции ядра CUDA, которая записывает результ копирования данных в PBO (в константную память).
/// </summary>
/// <param name="invViewMatrix">Матрица обратного вида.</param>
/// <param name="sizeofMatrix">Размер матрицы.</param>
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

/// <summary>
/// Инициализация буфера PBO.
/// </summary>
void initPixelBuffer();

/// <summary>
/// Рендеринг изображения, используя CUDA.
/// </summary>
void render()
{
    copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);               // Запись результа копирования данных в PBO (в константную память). 

    uint* devOutput;                                                    // Объявление указателя данных на CUDA-указатель.
    size_t nBytes;                                                      // Объявление переменной для хранения размера данных указателя.
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaResourcePBO, 0));  // Отображение данных PBO в указатель CUDA-девайса.

    // Получение указателя, с которым сможет работать CUDA.
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&devOutput, &nBytes, cudaResourcePBO));

    checkCudaErrors(cudaMemset(devOutput, 0, width * height * sizeof(float)));  // Обнуление данных PBO.

    // Вызов ядра CUDA для записи результата рендеринга в PBO.
    renderKernel(gridSize, blockSize, devOutput, width, height, density, brightness, transferOffset, transferScale);

    getLastCudaError("kernel failed");                                      // Вывод последней ошибки CUDA.
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaResourcePBO, 0));    // Завершение отображения ресурса в память CUDA.
}

/// <summary>
/// Функция отображения результатов с помощью OpenGL (вызывается GLUT).
/// </summary>
void display()
{
    sdkStartTimer(&timer);                          // старт таймера.

    GLfloat modelView[16];                          // Построение матрицы представления с использованием OpenGL.
    glMatrixMode(GL_MODELVIEW);                     // установка текущей матрицы ​в качестве "матрицы представления модели".
    glPushMatrix();                                 // сохранение матрицы, для последующего использования.                          
    glLoadIdentity();                               // преобразование текущей матрицы в единичную.                           
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);      // преобразование текущей матрицы матрицей поворота. 
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);   // преобразование текущей матрицы на матрицу перемащения.
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);    // получение готовой матрицы из OGL.
    glPopMatrix();                                  // восстановление последней сохраненной матриц (удаление из стека).

    // Зполнение транспонированной (инвертированной) матрицы значениями
    //////////////////////////////////////////
    // |00|01|02|03|        |00|01|02|03|   //              |00|04|08|12|
    // |04|05|06|07|    <=  |04|05|06|07|   //      =>      |01|05|09|13|
    // |08|09|10|11|        |08|09|10|11|   //              |02|06|10|10|
    //                      |12|13|14|15|   //
    //////////////////////////////////////////

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    
    render();                                   // Рендеринг изображения, используя CUDA.

    glClear(GL_COLOR_BUFFER_BIT);               // Очистка текущего буфера.
    glDisable(GL_DEPTH_TEST);                   // Деактивизация процедур буфера глубины. 
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);      // Установка свойства распаковки данных: режима выравнивания данных.

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);  // Копирование данных из PBO в текстуру.

    glBindTexture(GL_TEXTURE_2D, tex);          // Привязка текущей текстуры к буферу OpenGL.

    // Заполнение буфера данными.
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);    // Отвязка буфера от OpenGL.

    // Процесс отрисовки текстурированного четырехугольника (квада).
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();      // Смена переднего и заднего буферов для создания эффекта анимированного изображения.
    glutReportErrors();     // Вывод ошибки run-time выполнения OpenGL (в целях отладки).

    sdkStopTimer(&timer);   // Остановки таймера.
}

/// <summary>
/// Функция, отвечающая за простой программы.
/// </summary>
void idle()
{
    glutPostRedisplay();    // Вызов перерисовки текущего окна.
}

/// <summary>
/// Функция обработки взаимодействия с клавиатурой.
/// </summary>
/// <param name="key">Ключ нажатой клавиши.</param>
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:    // При нажатии на клавишу "ESC": закрытие окна и завершение работы программы.
            glutDestroyWindow(glutGetWindow());
            return;
            break;
        case 'f':   // При нажатии на клавишу "F": включение/отключение режима фильтрации (линейной интерполяции).
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;
        case '+':   // При нажатии на клавишу "+": увеличение плотности (путем изменения прозрачности).
            density += 0.01f;
            break;
        case '-':   // При нажатии на клавишу "-": уменьшение плотности (путем изменения прозрачности).
            density -= 0.01f;
            break;
        case ']':   // При нажатии на клавишу "]": увеличение яркости.
            brightness += 0.1f;
            break;
        case '[':   // При нажатии на клавишу "[": уменьшение яркости.
            brightness -= 0.1f;
            break;
        case ';':   // При нажатии на клавишу ";": увеличение смещения (по сути сила проникновения луча).
            transferOffset += 0.01f;
            break;
        case '\'':  // При нажатии на клавишу "'": уменьшение смещения (по сути сила проникновения луча).
            transferOffset -= 0.01f;
            break;
        case '.':   // При нажатии на клавишу ".": увеличение масштаба объекта.
            transferScale += 0.01f;
            break;
        case ',':   // При нажатии на клавишу ",": уменьшение масштаба объекта.
            transferScale -= 0.01f;
            break;
        default:
            break;
    }

    printf("density = %.2f, brightness = %.2f, transferOffset = %.2f, transferScale = %.2f\n", density, brightness, transferOffset, transferScale);
    glutPostRedisplay();    // Вызов перерисовки текущего окна.
}

int ox, oy;             // Координаты положения курсора на экране.
int buttonState = 0;    // Состояние кнопки.

/// <summary>
/// Функция обработки взаимодействия с компьютерной мышью.
/// </summary>
/// <param name="button">Идентификатор кнопки.</param>
/// <param name="state">Состояние кнопки.</param>
/// <param name="x">Координата положения курсора мыши на экране по X.</param>
/// <param name="y">Координата положения курсора мыши на экране по Y.</param>
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)         // Если состояние кнопки мыши - нажатое, то установка состояния мыши в глобальную переменную.
    {
        buttonState |= 1 << button;
    }
    else if (state == GLUT_UP)      // Если состояние кнопки мыши - нажатое, то сброс состояния мыши в глобальной переменной.
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();    // Вызов перерисовки текущего окна.
}

/// <summary>
/// Функция обработки движения курсора мыши по экрану.
/// </summary>
/// <param name="x">Координата положения курсора мыши на экране по X.</param>
/// <param name="y">Координата положения курсора мыши на экране по Y.</param>
void motion(int x, int y)
{
    // Вычисление разницы текущей и прошлой позиций курсора на экране 
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)           // Если состояние мыши 4 (нажата ПКМ): приближение/отдаление объекта.
    {
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)      // Если состояние мыши 2 (нажата СКМ (колесе)): перемещение объекта вдоль осей X и Y.
    {
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)      // Если состояние мыши 1 (нажата ЛКМ): Вращение объекта по осям X и Y.
    {
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();    // Высов перерисовки текущего окна.
}

/// <summary>
/// Функция определения количества блоков размером b для сеток размером a. 
/// </summary>
/// <param name="a">Размер сетки.</param>
/// <param name="b">Размер блока.</param>
/// <returns>Возвращает количество нужных блоков по одной оси.</returns>
int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

/// <summary>
/// Функция изменения размера окна отображения.
/// </summary>
/// <param name="w">Ширина окна.</param>
/// <param name="h">Длина окна.</param>
void reshape(int w, int h)
{
    width = w;
    height = h;
    initPixelBuffer();  // Инициализация буфера PBO.

    // Вычисление нового размера сетки.
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // Построение сетки, матриц и загрузка данных.
    glViewport(0, 0, w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

/// <summary>
/// Очистка данных и освобожение памяти из CUDA и OpenGL.
/// </summary>
void cleanup()
{
    sdkDeleteTimer(&timer);     // Удаление таймера.
    freeCudaBuffers();          // Освобожение памяти на CUDA.

    if (pbo)                    // Если буфер PBO уже имеется, то происходит отвязка от CUDA и удаление буфера и текстуры на OpenGL.
    {
        cudaGraphicsUnregisterResource(cudaResourcePBO);
        glDeleteBuffers(1, &pbo);
        glDeleteTextures(1, &tex);
    }
    
    // Cброс всех данных профиля перед выходом из приложения.
    checkCudaErrors(cudaProfilerStop());
}

/// <summary>
/// Инициализация компонентов OpenGL.
/// </summary>
void initGL(int *argc, char **argv)
{
    // Инициализация компонентов OpenGL
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Volume Rendering");

    // Проверка поддержки расширения PBO
    if (!isGLVersionSupported(2,0) || !areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
    {
        std::cout << "Required OpenGL extensions are missing." << std::endl;
        exit(EXIT_SUCCESS);
    }
}

/// <summary>
/// Инициализация буфера PBO.
/// </summary>
void initPixelBuffer()
{
    if (pbo)    // Если уже имеется объект OpenGL PBO.
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cudaResourcePBO));   // Удаление ресурса PBO из CUDA.
        glDeleteBuffers(1, &pbo);                                           // Удаление старого буфера PBO.
        glDeleteTextures(1, &tex);                                          // Удаление старой текстуры.
    }

    // Создание нового буфера PBO для вывода.
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // Регистрация данного PBO в качестве CUDA-ресурса.
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaResourcePBO, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Создание текстуры для вывода.
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/// <summary>
/// Загрузка данных файла с диска.
/// </summary>
/// <param name="filename"></param>
/// <param name="size"></param>
/// <returns></returns>
void *loadRawFile(char *filename, size_t size)
{
    FILE *fp = fopen(filename, "rb");       // открытие файла.
    if (!fp)                                // Если не удалось открыть файл.
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);              // выделение памяти
    size_t read = fread(data, 1, size, fp); // чтение файла
    fclose(fp);                             // закрытия потока работы с файлом

    printf("Read '%s', %zu bytes\n", filename, read);
    return data;
}

/// <summary>
/// Стартовый метод программы.
/// </summary>
/// <param name="argc">Количество аргументов командной строки.</param>
/// <param name="argv">Аргументы командной строки.</param>
/// <returns>Код ошибки или 0.</returns>
int main(int argc, char **argv)
{
    // Считывание аргуметов командной строки, полученных на вход программе
    pArgc = &argc;
    pArgv = argv;

    std::cout << "Starting of Volume Rendering\n" << std::endl;
    
    initGL(&argc, argv);                                        // Инициализация компонентов OpenGL.
    findCudaDevice(argc, (const char **)argv);                  // Поиск наилучшего CUDA устройства.
    char *filePath = sdkFindFilePath(volumeFilename, argv[0]);  // Получение относительного пути считываемого файла.

    if (filePath == 0)
    {
        std::cout << "Error finding file '" << volumeFilename << "'\n" << std::endl;
        exit(EXIT_FAILURE);
    }

    // определение размера данных
    size_t size = volumeSize.width * volumeSize.height * volumeSize.depth * sizeof(VolumeType);

    void *hVolume = loadRawFile(filePath, size);        // Получение данных из файла.
    initCuda(hVolume, volumeSize);                      // Инициализация CUDA-массивов и их настройка.
    free(hVolume);                                      // освобождение памяти
    sdkCreateTimer(&timer);                             // создание таймера

    std::cout << "Press '+' and '-' to change density (0.01 increments)" << std::endl;
    std::cout << "      ']' and '[' to change brightness" << std::endl;
    std::cout << "      ';' and ''' to modify transfer function offset" << std::endl;
    std::cout << "      '.' and ',' to modify transfer function scale\n" << std::endl;

    // вычисление размера сетки
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // Путь рендеринга для Volume Render.
    glutDisplayFunc(display);       // Регистрация обработчика отрисовки экрана.
    glutKeyboardFunc(keyboard);     // Регистрация обработчика взаимодействия с клавиатурой.
    glutMouseFunc(mouse);           // Регистрация обработчика взаимодействия с компютерной мышкой.
    glutMotionFunc(motion);         // Регистрация обработчика изменения позиции курсора мыши на экране. 
    glutReshapeFunc(reshape);       // Регистрация обработчика изменения размера окна.
    glutIdleFunc(idle);             // Регистрация обработчика простаивания программы.
    initPixelBuffer();              // Регистрация обработчика изменения размеров PBO (пересоздание PBO).
    glutCloseFunc(cleanup);         // Регистрация обработчика очистки памяти с CUDA и OpenGL.
    glutMainLoop();                 // Зацикливание выполения обработчиков событий.
}
