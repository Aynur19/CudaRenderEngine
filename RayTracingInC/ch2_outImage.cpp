#include <iostream>
#include <fstream>
#include "ch2_outImage.h"

using namespace std;


const int imgWidth = 512;
const int imgHeight = 512;
const wchar_t* imgFilename = L"image.ppm";

int outImage()
{
	ofstream imgOut;
	imgOut.open(imgFilename);

	imgOut << "P3\n" << imgWidth << " " << imgHeight << "\n255\n";

	for (int i = imgHeight - 1; i >= 0; --i)
	{
		for (int j = 0; j < imgWidth; ++j)
		{
			auto r = double(j) / (imgWidth - 1);
			auto g = double(i) / (imgHeight - 1);
			auto b = 0.25;

			int ir = static_cast<int>(255.999 * r);
			int ig = static_cast<int>(255.999 * g);
			int ib = static_cast<int>(255.999 * b);

			imgOut << ir << " " << ig << " " << ib << "\n";
		}
	}

	return 0;
}
