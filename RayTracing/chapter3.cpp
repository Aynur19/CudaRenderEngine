#include "chapter3.h"

int renderCPU_Ch3(Img img)
{
	std::ofstream imgOut;
	imgOut.open(img.cImgFilename);
	imgOut << "P3\n" << img.width << ' ' << img.height << "\n255\n";

    for (int j = img.height - 1; j >= 0; --j)
    {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < img.width; ++i)
        {
            color pColor(double(i) / (img.width - 1), double(j) / (img.height - 1), 0.25);
            writeColor(imgOut, pColor);
        }
    }

    imgOut.close();
    std::cerr << "\nDone.\n";
    return 0;
}

int runTest_Ch3()
{
    Img img = {};
    Block block = {};

    img.cImgFilename = L"output/ch3_cImage.ppm";
    img.cuImgFilename = L"output/ch3_cuImage.ppm";

    renderCPU_Ch3(img);
    std::cout << std::endl;
    renderGPU_Ch3(img, block);

    return 0;
}