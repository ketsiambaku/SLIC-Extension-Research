#include <vector>

enum SuperDuperPixelMode
{
	AVERAGE = 0,
	HISTOGRAM = 1
};

class SuperDuperPixel
{
public:
	SuperDuperPixel(int superpixel, std::vector<float> average, int pixel_count);
	SuperDuperPixel(int superpixel, std::vector< std::vector<float> >, int pixel_count);
	SuperDuperPixelMode get_mode();
	std::vector<int> get_superpixels();
	float distance_from(const std::vector<float>& average);
	float distance_from(const std::vector< std::vector<float> >& histogram);
	void add_superpixel(int superpixel, const std::vector<float>& average, int pixel_count);
	void add_superpixel(int superpixel, const std::vector< std::vector<float> >& histogram, int pixel_count);
	void operator+=(const SuperDuperPixel* other);
private:
	std::vector<int> superpixels;
	std::vector<float> average;
	// First dimension is color channel
	// Second dimension is bucket
	std::vector< std::vector<float> > histogram;
	int pixel_count;
	SuperDuperPixelMode mode;

	void add_average_superduperpixels(const SuperDuperPixel* other);
	void add_histogram_superduperpixels(const SuperDuperPixel* other);
};
