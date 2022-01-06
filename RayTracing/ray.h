#pragma once
#include "vec3.h"

class ray
{
public:
	point3 orig;
	vec3 dir;

public:
	__host__ __device__ ray() {}
	__host__ __device__ ray(const vec3& origin, const vec3& direction)
		:orig(origin), dir(direction) {}

	__host__ __device__ point3 origin() const { return orig; }
	__host__ __device__ vec3 direction() const { return dir; }

	__host__ __device__ point3 at(float t) const
	{
		return orig + t * dir;
	}
};