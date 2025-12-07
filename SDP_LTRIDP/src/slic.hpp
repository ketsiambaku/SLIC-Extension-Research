/**
 * slic.hpp
 * @brief LTriDP-enhanced SLIC superpixel segmentation for MRI images
 * 
 * @author Ketsia Mbaku
 * 
 * This file defines the LTriDPSuperpixelSLIC class, which implements the improved SLIC
 * algorithm described in [1].
 * 
 * The implementation is adapted from OpenCV's SuperpixelSLIC [2]
 * with two key modifications:
 * 
 * 1. Texture-aware distance metric:
 *    D = sqrt((dc/Nc)² + (dt/Nt)² + (ds/Ns)²)
 *    where dc=gray distance, dt=LTriDP texture distance, ds=spatial distance
 * 
 * 2. Gray-difference threshold filtering for cluster center updates:
 *    Only pixels satisfying |gray_center - gray_pixel| < α are used
 *    to compute new cluster centers, preventing misclassified edge pixels
 *    from deforming superpixel boundaries.
 * 
 * References:
 * [1] Wang, Y., Qi, Q., & Shen, X. (2020). Image Segmentation of Brain MRI
 *     Based on LTriDP and Superpixels of Improved SLIC. Brain Sciences, 10(2), 116.
 * [2] OpenCV SLIC implementation: opencv_contrib/modules/ximgproc/src/slic.cpp
 */

#ifndef SLIC_HPP
#define SLIC_HPP

#include <opencv2/core.hpp>
#include <vector>
#include <set>
#include <list>
#include "superduperpixel.hpp"

namespace ltridp {

/**
 * @class LTriDPSuperpixelSLIC
 * @brief Texture-enhanced SLIC superpixel segmentation with gray-threshold center updating
 * 
 * This class implements the improved SLIC algorithm that incorporates LTriDP texture
 * features into the distance metric and uses gray-difference threshold filtering
 * when updating cluster centers.
 */
class LTriDPSuperpixelSLIC {
public:
    /**
     * @brief Constructor - Initialize a new LTriDPSuperpixelSLIC object
     * 
     * Pre-conditions:
     * - image must be non-empty, gray-scale CV_8U
     * - texture must be non-empty
     * - image and texture must have same dimensions
     * - region_size must be > 0 (usually: 10-30)
     * - ruler must be > 0 to control compactness (usually: 5.0-20.0)
     * 
     * Post-conditions:
     * - Object initialized and ready for iteration
     * - Cluster centers placed on regular grid
     * - Centers perturbed away from edges
     * 
     * @param image Enhanced grayscale MRI image
     * @param texture LTriDP texture feature map (0-255)              
     * @param region_size Approximate superpixel size S
     * @param ruler Compactness parameter m
     */
    LTriDPSuperpixelSLIC(const cv::Mat& image, const cv::Mat& texture, int region_size = 20, float ruler = 10.0f);
    
    /**
     * @brief Destructor - Clean up resources
     */
    ~LTriDPSuperpixelSLIC();
    
    /**
     * @brief Perform superpixel segmentation iterations
     * 
     * Pre-conditions:
     * Object properly initialized via constructor
     * num_iterations > 0
     * 
     * Post-conditions:
     * - Superpixel labels updated for all pixels
     * 
     * Algorithm per iteration:
     * 1. For each cluster center, search 2S×2S neighborhood
     * 2. Assign pixels to nearest center using new distance metric
     * 3. Update centers using gray-threshold filtered pixels
     * 4. Repeat until convergence or max iterations
     * 
     * @param num_iterations Number of k-means iterations (paper used 10)
     */
    void iterate(int num_iterations = 10);
    
    /**
     * @brief Get superpixel labels for each pixel
     * 
     * Pre-conditions:
     * - iterate() has been called at least once
     * 
     * Post-conditions:
     * - labels_out contains matrix (width × height)
     * - Each pixel value is superpixel ID in range [0, getNumberOfSuperpixels()-1]
     * 
     * @param labels_out Output label matrix
     */
    void getLabels(cv::Mat& labels_out) const;
    
    /**
     * @brief Get the number of superpixels generated
     * 
     * Post-conditions:
     * - Returns actual number of superpixels after iteration
     * - May differ from initial estimate due to boundary effects
     */
    int getNumberOfSuperpixels() const;
    
    /**
     * @brief Generate superpixel boundary contour mask
     * 
     * Pre-conditions:
     * - iterate() has been called at least once
     * 
     * Post-conditions:
     * - mask contains CV_8U binary image (width × height)
     * - Boundary pixels set to 255, interior pixels set to 0
     * 
     * @param mask Output boundary mask
     * @param thick_line If true, use 2-pixel boundary; if false, 1-pixel
     */
    void getLabelContourMask(cv::Mat& mask, bool thick_line = true) const;
    
    /**
     * @brief Enforce label connectivity by merging small disconnected regions
     * 
     * Post-conditions:
     * - Small disconnected components merged into adjacent superpixels
     * - Label matrix updated to ensure connectivity
     * 
     * @param min_element_size Minimum superpixel size as percentage
     */
    void enforceLabelConnectivity(int min_element_size = 25);

	//////////////////////////////////////////////////
	//
	// Custom methods
	//
	//////////////////////////////////////////////////
	
	/** @brief Combines adjacent superpixels into super-duper-pixels if they're similar enough in color.
	
	Uses average colors of superpixels to determine if they're similar enough in color.

    @param distance The max distance the average colors of superpixels can be from each other to be
	combined.
     */
	CV_WRAP virtual void duperizeWithAverage(const float distance) = 0;

	/** @brief Combines adjacent superpixels into super-duper-pixels if they're similar enough in color.
	
	Uses distances between (normalized) color histograms of superpixels to determine if they're similar
	enough in color.

    @param num_buckets The number of histogram buckets to use for each color channel
	(RGB, HSV, LAB, etc.).

	@param distance The max distance the (normalized) color histograms of superpixels can be from each
	other to be combined.
     */
	CV_WRAP virtual void duperizeWithHistogram(const int num_buckets[], const float distance) = 0;

protected:
    // Image dimensions
    int m_width;         
    int m_height;       
    
    // Algorithm parameters
    int m_region_size;     // Superpixel size S (grid interval)
    float m_ruler;         // Compactness parameter m
    
    // Superpixel state
    int m_numlabels;       // Number of superpixels
    cv::Mat m_klabels;     // Label matrix (CV_32S, width × height)
    
    // Input images
    cv::Mat m_image;       // Enhanced grayscale image (CV_8U)
    cv::Mat m_texture;     // LTriDP texture features (CV_8U)   // ADDED
    
    // Cluster centers
    std::vector<float> m_kseedsx;      // Cluster center x-coordinates
    std::vector<float> m_kseedsy;      // Cluster center y-coordinates
    std::vector<float> m_kseeds_gray;  // Cluster center gray values
    std::vector<float> m_kseeds_tex;   // Cluster center texture values   // ADDED: texture centers

private:
    /**
     * @brief Initialize cluster centers on regular grid and perturb away from edges
     * 
     * Algorithm:
     * 1. Calculate number of superpixels: K = (width * height) / (S * S)
     * 2. Place centers on regular grid with spacing S
     * 3. Detect edges
     * 4. Move each center to lowest-gradient position in 3×3 neighborhood
     */
    void initialize();
    
    /**
     * @brief Detect edges in the image.
     * 
     * @param edgemag Output edge magnitude map (CV_32F)
     */
    void detectEdges(cv::Mat& edgemag);
    
    /**
     * @brief Perturb seeds away from high-gradient (edge) locations
     * 
     * For each seed, searches 3×3 neighborhood and moves to position
     * with minimum edge magnitude.
     * 
     * @param edgemag Edge magnitude map (CV_32F)
     */
    void perturbSeeds(const cv::Mat& edgemag);
    
    /**
     * @brief Get seeds on regular grid with spacing = region_size
     * 
     * Creates initial cluster centers placed uniformly across image
     * in a grid pattern with spacing S (region_size).
     */
    void getSeeds();
    
    /**
     * @brief Perform one iteration of the improved SLIC algorithm
     * 
     * Algorithm per iteration:
     * 1. Initialize distance matrix to infinity
     * 2. For each cluster center k:
     *    a. Search 2S×2S neighborhood around center
     *    b. For each pixel i in neighborhood:
     *       - Compute gray distance: dc = |gray_k - gray_i|
     *       - Compute texture distance: dt = |texture_k - texture_i|        // ADDED: texture distance
     *       - Compute spatial distance: ds = sqrt((x_k-x_i)² + (y_k-y_i)²)
     *       - Compute combined distance: D = sqrt((dc/Nc)² + (dt/Nt)² + (ds/Ns)²)  // ADDED: dt term
     *       - If D < distance[i], assign pixel i to cluster k
     * 3. Update cluster centers using gray-threshold filtering (see updateCenters)
     * 
     * @param num_iterations Total number of iterations
     */
    void performLTriDPSLIC(int num_iterations);
    
    /**
     * @brief Update cluster centers using gray-difference threshold filtering
     * 
     * This is the key modification from the paper. Instead of averaging all pixels
     * in a cluster, only pixels with gray difference below threshold α contribute
     * to the new center position.
     * 
     * Algorithm:
     * For each cluster k:
     * 1. Let gray_k = current cluster center gray value
     * 2. For each pixel i assigned to cluster k:
     *    - If |gray_k - gray_i| < α, include pixel in average
     * 3. Compute new center as average of included pixels:
     *    - x_k = mean of included x-coordinates
     *    - y_k = mean of included y-coordinates
     *    - gray_k = mean of included gray values
     *    - texture_k = mean of included texture values
     * 
     * The threshold α is computed as the standard deviation of image gray values.
     */
    void updateCenters();
    
    /**
     * @brief Calculate gray-difference threshold α (standard deviation of image)
     * 
     * Pre-conditions:
     * - m_image is valid CV_8U matrix
     * 
     * Post-conditions:
     * - Returns standard deviation of gray values across entire image
     * 
     * @return Gray threshold α for center update filtering
     */
    float calculateGrayThreshold() const;  // ADDED: threshold calculation

	//////////////////// Custom Methods ////////////////////

	inline void findSuperpixelNeighborsAndAverages
	(
		std::vector< std::set<int> >& superpixel_neighbors,
		std::vector< std::vector<float> >& superpixel_average_colors,
		std::vector<int>& superpixel_population
	);

	inline void findSuperpixelNeighborsAndHistograms
	(
		const int num_buckets[],
		std::vector< std::set<int> >& superpixel_neighbors,
		std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
		std::vector<int>& superpixel_population
	);

	inline void addColorsToAverages
	(
		std::vector< std::vector<float> >& superpixel_average_colors,
		const int current_superpixel,
		const int x,
		const int y
	);

	inline void addColorsToHistograms
	(
		const int num_buckets[],
		std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
		const int current_superpixel,
		const int x,
		const int y
	);

	inline void linkNeighborSuperpixels
	(
		std::vector< std::set<int> >& superpixel_neighbors,
		const int current_superpixel,
		const int x,
		const int y
	);

	inline void groupSuperpixels
	(
		const float max_distance,
		const std::vector< std::set<int> >& superpixel_neighbors,
		const std::vector< std::vector<float> >& superpixel_average_colors,
		const std::vector<int>& superpixel_population,
		std::list<SuperDuperPixel>& superduperpixels,
		std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators
	);

	inline void groupSuperpixels
	(
		const int num_buckets[],
		const float max_distance,
		const std::vector< std::set<int> >& superpixel_neighbors,
		const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
		const std::vector<int>& superpixel_population,
		std::list<SuperDuperPixel>& superduperpixels,
		std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators
	);

	inline float getColorDistance
	(
		const std::list<SuperDuperPixel>& superduperpixels,
		const std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		const std::vector< std::vector<float> >& superpixel_average_colors, 
		std::vector<float>& average_colors,
		const int superpixel,
		const int neighbor
	);

	inline float getColorDistance
	(
		const int num_buckets[],
		const std::list<SuperDuperPixel>& superduperpixels,
		const std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms, 
		std::vector< std::vector<float> >& color_histogram,
		const int superpixel,
		const int neighbor
	);

	inline void extractAverageColors
	(
		const std::vector< std::vector<float> >& superpixel_average_colors,
		std::vector<float>& average_colors,
		const int superpixel
	);

	inline void extractColorHistogram
	(
		const int num_buckets[],
		const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
		std::vector< std::vector<float> >& color_histogram,
		const int superpixel
	);

	inline void combineIntoSuperDuperPixel
	(
		std::list<SuperDuperPixel>& superduperpixels,
		std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators,
		const std::vector< std::vector<float> >& superpixel_average_colors,
		const std::vector<float>& average_colors,
		const std::vector<int>& superpixel_population,
		const int superpixel,
		const int neighbor
	);

	inline void combineIntoSuperDuperPixel
	(
		const int num_buckets[],
		std::list<SuperDuperPixel>& superduperpixels,
		std::vector<SuperDuperPixel*>& superduperpixel_pointers,
		std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators,
		const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
		const std::vector< std::vector<float> >& color_histogram,
		const std::vector<int>& superpixel_population,
		const int superpixel,
		const int neighbor
	);

	inline int indexSuperduperpixels
	(
		const std::list<SuperDuperPixel>& superduperpixels,
		std::vector<int>& superduperpixel_indexes
	);

	inline void assignSuperduperpixels(const std::vector<int>& superduperpixel_indexes);

	static const int m_nr_channels = 1;

	//////////////////// Custom Methods ////////////////////
};

} // namespace ltridp

#endif // SLIC_HPP
