#pragma once
#include "vec3.h"


class camera
{
public:
	float viewportHeight = 1.0;
	float viewportWidth = 1.0;
	float focalLength = 1.0;

	point3 origin = point3(0, 0, 0);
	vec3 horizontal = vec3(0, 0, 0);
	vec3 vertical = vec3(0, 0, 0);
	vec3 lowerLeftCorner = vec3(0, 0, 0);

public:
	camera() {};
};