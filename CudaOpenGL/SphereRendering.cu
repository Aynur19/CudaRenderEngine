#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>

#define DIM 600
#define rnd(x)(x*rand()/RAND_MAX)
#define INF 2e10f

struct Sphere
{
	float r, g, b;
	float radius;
	float x, y, z;


	__device__ float hit(float ox, float oy, float* n)
	{
		float dx2 = (ox - x) * (ox - x);
		float dy2 = (oy - y) * (oy - y);
		float rad2 = radius * radius;

		if (dx2 + dy2 < rad2)
		{
			float dz = sqrtf(rad2 - dx2 - dy2);
			*n = dz / radius;
			return dz + z;
		}

		return -INF;
	}
};



void main()
{
	CPUBitmap
}
