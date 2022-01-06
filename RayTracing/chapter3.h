#pragma once
#include <iostream>
#include <fstream>

#include "structs.h"
#include "vec3.h"
#include "color.h"

int renderCPU_Ch3(Img img);

int renderGPU_Ch3(Img img, Block block);

int runTest_Ch3();