#pragma once
#include <iostream>
#include <fstream>

#include "vec3.cuh"

template<typename T>
void writeColor(T& out, color pColor)
{
    int ir = int(255.999 * pColor.r());
    int ig = int(255.999 * pColor.g());
    int ib = int(255.999 * pColor.b());

    out << ir << ' ' << ig << ' ' << ib << '\n';
}

