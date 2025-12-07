#include "SuperDuperPixel.hpp"
#include <assert.h>

SuperDuperPixel::SuperDuperPixel(int superpixel, std::vector<float> average, int pixel_count)
{
	this->superpixels.push_back(superpixel);
	this->average = average;
	this->pixel_count = pixel_count;
	this->mode = AVERAGE;
}

SuperDuperPixel::SuperDuperPixel(int superpixel, std::vector< std::vector<float> > histogram, int pixel_count)
{
	this->superpixels.push_back(superpixel);
	this->histogram = histogram;
	this->pixel_count = pixel_count;
	this->mode = HISTOGRAM;
}

SuperDuperPixelMode SuperDuperPixel::get_mode() { return this->mode; }
std::vector<int> SuperDuperPixel::get_superpixels() { return this->superpixels; }

float SuperDuperPixel::distance_from(const std::vector<float>& average)
{
	assert(this->average.size() == average->size());
	float dist = 0;
	for (int color_channel = 0; color_channel < this->average.size(); color_channel += 1)
	{
		float diff = this->average[color_channel] - average[color_channel];
		// OpenCV SLIC algorithm square diff before adding it to dist.
		// dist += diff * diff;
		// Just take absolute value to do mahnattan distance instead.
		dist += abs(diff);
	}
	// Just use manhattan distance here.
	// Could do this to be more precise (euclidian distance, would also need to square the diff above), but OpenCV
	// SLIC algorithm doesn't use it either.
	// dist = sqrt(dist);
	return dist;
}

float SuperDuperPixel::distance_from(const std::vector< std::vector<float> >& histogram)
{
	assert(this->histogram.size() == histogram->size());
	float dist = 0;
	for (int color_channel = 0; color_channel < this->histogram.size(); color_channel += 1)
	{
		assert(this->histogram[color_channel].size() == histogram[color_channel].size());
		for (int bucket = 0; bucket < this->histogram[color_channel].size(); color_channel += 1)
		{
			float diff = this->histogram[color_channel][bucket] - histogram[color_channel][bucket];
			// OpenCV SLIC algorithm square diff before adding it to dist.
			// dist += diff * diff;
			// Just take absolute value to do mahnattan distance instead.
			dist += abs(diff);
		}
	}
	// Just use manhattan distance here.
	// Could do this to be more precise (euclidian distance, would also need to square the diff above), but OpenCV
	// SLIC algorithm doesn't use it either.
	// dist = sqrt(dist);
	return dist;
}

void SuperDuperPixel::add_superpixel(int superpixel, const std::vector<float>& average, int pixel_count)
{
	assert(this->average.size() == average.size());
	this->superpixels.push_back(superpixel);
	int new_pixel_count = this->pixel_count + pixel_count;
	for (int color_channel = 0; color_channel < this->average.size(); color_channel += 1)
	{
		float this_sum = this->average[color_channel] * this->pixel_count;
		float other_sum = average[color_channel] * pixel_count;
		this->average[color_channel] = (this_sum + other_sum) / new_pixel_count;
	}
	this->pixel_count = new_pixel_count;
}

void SuperDuperPixel::add_superpixel(int superpixel, const std::vector< std::vector<float> >& histogram, int pixel_count)
{
	assert(this->histogram.size() == histogram.size());
	this->superpixels.push_back(superpixel);
	int new_pixel_count = this->pixel_count + pixel_count;
	for (int color_channel = 0; color_channel < this->histogram.size(); color_channel += 1)
	{
		assert(this->histogram[color_channel].size() == histogram[color_channel].size());
		for (int bucket = 0; bucket < this->histogram[color_channel].size(); bucket += 1)
		{
			float this_sum = this->histogram[color_channel][bucket] * this->pixel_count;
			float other_sum = histogram[color_channel][bucket] * pixel_count;
			this->histogram[color_channel][bucket] = (this_sum + other_sum) / new_pixel_count;
		}
	}
	this->pixel_count = new_pixel_count;
}

void SuperDuperPixel::operator+=(const SuperDuperPixel* other)
{
	assert(this->mode == other->mode);
	this->superpixels.insert(this->superpixels.end(), other->superpixels.begin(), other->superpixels.end());
	switch (this->mode)
	{
		case AVERAGE:
			this->add_average_superduperpixels(other);
			break;
		case HISTOGRAM:
			this->add_histogram_superduperpixels(other);
	}
}

void SuperDuperPixel::add_average_superduperpixels(const SuperDuperPixel* other)
{
	assert(this->average.size() == other->average.size());
	int new_pixel_count = this->pixel_count + other->pixel_count;
	for (int color_channel = 0; color_channel < this->average.size(); color_channel += 1)
	{
		float this_sum = this->average[color_channel] * this->pixel_count;
		float other_sum = other->average[color_channel] * other->pixel_count;
		this->average[color_channel] = (this_sum + other_sum) / new_pixel_count;
	}
	this->pixel_count = new_pixel_count;
}

void SuperDuperPixel::add_histogram_superduperpixels(const SuperDuperPixel* other)
{
	assert(this->histogram.size() == other->histogram.size());
	int new_pixel_count = this->pixel_count + other->pixel_count;
	for (int color_channel = 0; color_channel < this->average.size(); color_channel += 1)
	{
		assert(this->histogram[color_channel].size() == other->histogram[color_channel].size());
		for (int bucket = 0; bucket < this->histogram[color_channel].size(); bucket += 1)
		{
			float this_sum = this->histogram[color_channel][bucket] * this->pixel_count;
			float other_sum = other->histogram[color_channel][bucket] * other->pixel_count;
			this->histogram[color_channel][bucket] = (this_sum + other_sum) / new_pixel_count;
		}
	}
	this->pixel_count = new_pixel_count;
}
