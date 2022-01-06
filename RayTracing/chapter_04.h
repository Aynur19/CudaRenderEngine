#pragma once
#include <iostream>
#include <fstream>

#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "structs.h"
#include "camera.h"

int ch04_renderCPU(Img img, camera& cam);

int ch04_runTest();