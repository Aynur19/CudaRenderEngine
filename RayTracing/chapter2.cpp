#include "chapter2.h"

int renderCPU_Ch2(Img img)
{
	std::ofstream imgOut;
	imgOut.open(img.cImgFilename);
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

int runTest_Ch2()
{
	Img img = {};
	Block block = {};

	img.cImgFilename = L"output/ср2_cImage.ppm";
	img.cuImgFilename = L"output/ср2_cuImage.ppm";

	renderCPU_Ch2(img);
	std::cout << std::endl;
	renderGPU_Ch2(img, block);

	return 0;
}