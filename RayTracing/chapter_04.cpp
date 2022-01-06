#include "chapter_04.h"

color rayColor(const ray& r)
{
	vec3 unitDirection = getUnitVector(r.direction());
	auto t = 0.5 * (unitDirection.y() + 1.0);

	return(1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int ch04_renderCPU(Img img, camera& cam)
{
	// Render
	std::ofstream imgOut;
	imgOut.open(img.cImgFilename);
	imgOut << "P3\n" << img.width << " " << img.height << "\n255\n";

	for (int j = img.height - 1; j >= 0; --j)
	{
		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
		for (int i = 0; i < img.width; ++i)
		{
			auto u = double(i) / (img.width - 1);
			auto v = double(j) / (img.height - 1);
			auto direction = cam.lowerLeftCorner + u * cam.horizontal + v * cam.vertical - cam.origin;
			ray r(cam.origin, direction);
			color pixel_color = rayColor(r);
			writeColor(imgOut, pixel_color);
		}
	}
	imgOut.close();
	std::cerr << "\nDone.\n";
	return 0;
}

int ch04_runTest()
{
	// Image
	Img img = {};
	img.cImgFilename = L"output/ch04_cImage.ppm";
	
	img.aspectRatio = 16.0 / 9.0;
	img.width = 1200;
	img.height = int(img.width / img.aspectRatio);

	// Camera
	auto cam = camera();
	cam.viewportHeight = 2.0;
	cam.viewportWidth = img.aspectRatio * cam.viewportHeight;
	cam.focalLength = 1.0;

	cam.origin = point3(0, 0, 0);
	cam.horizontal = vec3(cam.viewportWidth, 0, 0);
	cam.vertical = vec3(0, cam.viewportHeight, 0);
	cam.lowerLeftCorner = cam.origin - cam.horizontal / 2 - cam.vertical / 2 - vec3(0, 0, cam.focalLength);

	ch04_renderCPU(img, cam);
	
	return 0;
}
