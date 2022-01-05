#include <iostream>
#include <fstream>

#include "chapter2.h"

using namespace std;

int renderCPU(const wchar_t* imgFilename, Img img)
{
	ofstream imgOut;
	imgOut.open(imgFilename);
	imgOut << "P3\n" << img.width << " " << img.height << "\n255\n";

	for (int i = img.height - 1; i >= 0; --i)
	{
		std::cerr << "\Scanlines remaining: " << i << "\n" << std::flush;
		for (int j = 0; j < img.width; ++j)
		{
			auto r = double(j) / (img.width - 1);
			auto g = double(i) / (img.height - 1);
			auto b = 0.25;

			int ir = static_cast<int>(255.999 * r);
			int ig = static_cast<int>(255.999 * g);
			int ib = static_cast<int>(255.999 * b);

			imgOut << ir << " " << ig << " " << ib << "\n";
		}
	}

	imgOut.close();
	std::cerr << "Done.\n";

	return 0;
}

int runTest2()
{
	const int imgWidth = 512;
	const int imgHeight = 512;

	const wchar_t* cImgFilename = L"cImage.ppm";
	const wchar_t* cuImgFilename = L"cuImage.ppm";

	Img img = {};
	Block block = {};

	renderCPU(cImgFilename, img);
	std::cout << std::endl;
	renderGPU(cuImgFilename, img, block);

	return 0;
}