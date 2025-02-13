#include "CTRL_Bmp.h"


void BMP_static::save(const string& file_name, const vector<RGB>& my_pixel)
{
    u32 width = def_WIDTH;
    u32 height = def_HEIGHT;

	ofstream f;
	f.open(file_name.c_str(), std::ios::out | std::ios::binary);	

	if (f.fail())
	{
		std::cout << "BMP_img could not open file\n";
		return;
	}

	unsigned char padding[3] = { 0, 0, 0 };
	const int size_padding = ((4 - (width * 3) % 4) % 4);

	const int size_header = 14;
	const int size_info_header = 40;
	const int size_file = size_header + size_info_header
						+ (3 * width * height)
						+ (size_padding * height);

	unsigned char header[size_header] {};
	unsigned char info_header[size_info_header] {};


	// HEADER //

	// Symbol
	header[0] = 'B';
	header[1] = 'M';
	// File size
	header[2] = size_file;
	header[3] = size_file >> 8;
	header[4] = size_file >> 16;
	header[5] = size_file >> 24;
	// Reserved
	header[6] = 0;
	header[7] = 0;
	header[8] = 0;
	header[9] = 0;
	// Data offset
	header[10] = size_header + size_info_header;
	header[11] = 0;
	header[12] = 0;
	header[13] = 0;


	// INFO HEADER //

	// Header size
	info_header[0] = size_info_header;
	info_header[1] = 0;
	info_header[2] = 0;
	info_header[3] = 0;
	// Width
	info_header[4] = width;
	info_header[5] = width >> 8;
	info_header[6] = width >> 16;
	info_header[7] = width >> 24;
	// Height
	info_header[8] = height;
	info_header[9] = height >> 8;
	info_header[10] = height >> 16;
	info_header[11] = height >> 24;
	// Planes
	info_header[12] = 1;
	info_header[13] = 0;
	// Bits per pixel
	info_header[14] = 24;
	info_header[15] = 0;

	f.write(reinterpret_cast<char*>(header), size_header);
	f.write(reinterpret_cast<char*>(info_header), size_info_header);

	//for (int y = 0; y < height; y++)
	for (int y = height-1; y >= 0; y--)
	{
		for (int x = 0; x < width; x++)
		{
            RGB current_color = my_pixel[def_convert_2d_to_1d(x, y)];

			unsigned char COLOR[] =
            {
                current_color.b,
                current_color.g,
                current_color.r
            };
			f.write(reinterpret_cast<char*>(COLOR), 3);
		}

		f.write(reinterpret_cast<char*>(padding), size_padding);
	}

	f.close();
}