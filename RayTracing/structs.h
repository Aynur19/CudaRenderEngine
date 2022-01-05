#pragma once

struct Img
{
	wchar_t* cImgFilename = L"cImage.ppm";
	wchar_t* cuImgFilename = L"cuImage.ppm";

	int width = 1200;
	int height = 600;
};

struct Block
{
	int tx = 32;
	int ty = 32;
};