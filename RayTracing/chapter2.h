#pragma once

#include <iostream>
#include <fstream>

struct Img
{
	int width = 1200;
	int height = 600;
};

struct Block
{
	int tx = 32;
	int ty = 32;
};

int renderGPU(const wchar_t* imgFilename, Img img, Block block);

int renderCPU(const wchar_t* imgFilename, Img img);

int runTest2();