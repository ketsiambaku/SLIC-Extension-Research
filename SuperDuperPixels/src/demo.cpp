// Demo.cpp
// Demonstrates SD-SLIC (Super Duper - Simple Linear Iterative Clustering)
// Author: Chandler Calkins

#include <iostream>
#include <string>
#ifdef _WIN32
    #include <direct.h>
    #define chdir _chdir
#else
    #include <unistd.h>
#endif
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/ximgproc/slic.hpp>
#include "sdp_slic.hpp"
using namespace cv;

// Displays an image with superpixel borders overlayed onto it
// Inputs:
// slic: slic object that generated superpixels for input_image
// input_image: the image that superpixels were generated for using slic
// window_name: the name of the window to display the superpixels to using cv::imshow()
Mat show_superpixels(const Ptr<SuperpixelSLIC>& slic, const Mat& input_image, const String& window_name)
{
	// Gets overlay image of superpixels
	Mat superpixels;
	slic->getLabelContourMask(superpixels);

	// Creates the output image of superpixels
	Mat output = input_image.clone();
	// Set each pixel in output to white if it's a superpixel border
	for (int row = 0; row < output.rows; row += 1)
	for (int col = 0; col < output.cols; col += 1)
	{
		if (superpixels.at<uchar>(row, col) != 0)
		{
			output.at<Vec3b>(row, col)[0] = superpixels.at<uchar>(row, col);
			output.at<Vec3b>(row, col)[1] = superpixels.at<uchar>(row, col);
			output.at<Vec3b>(row, col)[2] = superpixels.at<uchar>(row, col);
		}
	}

	// Displays output to a window
	imshow(window_name, output);
	waitKey(0);
	return output;
}

// main - Generates superpixels for an images using SD-SLIC and displays those superpixels on the image.
//
// Preconditions:
//
// There is a valid .jpg .png or .gif file in the project folder and the `imread()` call that creates the `input_image` object reads from that file.
//
// Postconditions:
//
// A file called output.png should be in the project folder.
int main(int argc, char* argv[])
{
	// Move out of build/Debug into root of project folder
	// Use this for VSCode, comment out for Visual Studio / any setups where the input file is in the same folder as the executable
	chdir("../../");

	// Reads the input image
	Mat input_image;
	FILE* file;
	if (fopen_s(&file, "input.jpg", "r") == 0) input_image = imread("input.jpg");
	else if (fopen_s(&file, "input.png", "r") == 0) input_image = imread("input.png");
	else if(fopen_s(&file, "input.gif", "r") == 0) input_image = imread("input.gif");
	else
	{
		std::cerr << "ERROR: No input file found / accessible. This program needs an 'input.jpg' or 'input.png' file in the same folder to work.\n";
		return 1;
	}

	Mat cielab_image;
	cvtColor(input_image, cielab_image, COLOR_RGB2Lab);

	// Creates window to display output to
	const String window_name = "Superpixels";
	namedWindow(window_name);

	const int avg_superpixel_size = 100; // Default: 100
	const float smoothness = 100.0f; // Default: 10.0
	const int min_superpixel_size_percent = 4;
	// Ptr<ximgproc::SuperpixelSLIC> slic = ximgproc::createSuperpixelSLIC(input_image, ximgproc::SLIC, avg_superpixel_size, smoothness);
	Ptr<SuperpixelSLIC> slic = createSuperpixelSLIC(cielab_image, SLIC, avg_superpixel_size, smoothness);
	slic->iterate(1);
	slic->enforceLabelConnectivity(min_superpixel_size_percent);

	show_superpixels(slic, input_image, window_name);
	
	// 50.0 good for aguilles_rogues, 32.0 good for cosmo
	slic->duperizeWithAverage(30.0);
	// const int num_buckets[] = {16, 16, 16};
	// slic->duperizeWithHistogram(num_buckets, 2.0f);

	Mat output = show_superpixels(slic, input_image, window_name);
	
	// Write output to an image file
	imwrite("output.png", output);

	return 0;
}
