/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * SLIC, SLICO Copyright (c) 2013
 * Radhakrishna Achanta
 * email : Radhakrishna [dot] Achanta [at] epfl [dot] ch
 * web : http://ivrl.epfl.ch/people/achanta
 *
 * MSLIC Copyright (c) 2016, 2017
 * Yong-Jin Liu
 * email : liuyongjin [at] tsinghua [dot] edu [dot] cn
 * web : http://47.89.51.189/liuyj
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holders nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
/**
 * @file slic.cpp
 * @brief Implementation of improved SLIC superpixel segmentation
 * 
 * @author Ketsia Mbaku
 * 
 * This file implements the LTriDPSuperpixelSLIC class.
 * 
 * References:
 * [1] Wang et al. (2020), "Image Segmentation of Brain MRI Based on LTriDP 
 *   and Superpixels of Improved SLIC", Brain Sciences 10(2), 116.
 * [2] OpenCV SLIC: opencv_contrib/modules/ximgproc/src/slic.cpp
 */

#include "slic.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

namespace ltridp {

LTriDPSuperpixelSLIC::LTriDPSuperpixelSLIC(const cv::Mat& image,
                                           const cv::Mat& texture,  // ADDED: texture input
                                           int region_size,
                                           float ruler)
    : m_region_size(region_size), m_ruler(ruler)
{
    // Validate inputs
    if (image.empty()) {
        throw std::invalid_argument("Input image must not be empty");
    }
    if (texture.empty()) {                                
        throw std::invalid_argument("Texture image must not be empty");
    }
    if (image.type() != CV_8UC1) {
        throw std::invalid_argument("Input image must be CV_8UC1 (grayscale)");
    }
    if (texture.type() != CV_8UC1) {                                   
        throw std::invalid_argument("Texture image must be CV_8UC1");
    }
    if (image.size() != texture.size()) {                               
        throw std::invalid_argument("Image and texture must have same dimensions");
    }
    if (region_size <= 0) {
        throw std::invalid_argument("Region size must be positive");
    }
    if (ruler <= 0.0f) {
        throw std::invalid_argument("Ruler (compactness) must be positive");
    }
    
    // Store dimensions
    m_width = image.cols;
    m_height = image.rows;
    
    // Store input images (deep copy for safety)
    m_image = image.clone();
    m_texture = texture.clone();                                        // ADDED: store texture
    
    // Initialize superpixel segmentation
    initialize();
}

LTriDPSuperpixelSLIC::~LTriDPSuperpixelSLIC()
{
    // Automatic cleanup via std::vector and cv::Mat destructors
}

void LTriDPSuperpixelSLIC::initialize()
{
    // Calculate initial number of superpixels
    // K = N / S² where N = total pixels, S = region_size
    m_numlabels = static_cast<int>(
        static_cast<float>(m_width * m_height) / 
        static_cast<float>(m_region_size * m_region_size)
    );
    
    // Initialize label storage (all zeros)
    m_klabels = cv::Mat(m_height, m_width, CV_32S, cv::Scalar(0));
    
    // Generate initial seeds on regular grid
    getSeeds();
    
    // Update actual number of labels after seed generation
    m_numlabels = static_cast<int>(m_kseeds_gray.size());
    
    // Perturb seeds away from edges for better boundaries
    cv::Mat edgemag;
    detectEdges(edgemag);
    perturbSeeds(edgemag);
}

void LTriDPSuperpixelSLIC::detectEdges(cv::Mat& edgemag)
{
    // Compute gradient magnitude using Sobel operators
    // This follows OpenCV SLIC's DetectChEdges approach
    
    cv::Mat dx, dy;
    cv::Sobel(m_image, dx, CV_32F, 1, 0, 3);  // Horizontal gradient
    cv::Sobel(m_image, dy, CV_32F, 0, 1, 3);  // Vertical gradient
    
    // Compute magnitude
    cv::magnitude(dx, dy, edgemag);
}

void LTriDPSuperpixelSLIC::perturbSeeds(const cv::Mat& edgemag)
{
    // For each seed, search 3×3 neighborhood and move to
    // position with minimum edge magnitude to avoid placing seeds on edges
    
    const int dx8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    for (int k = 0; k < m_numlabels; ++k) {
        int cx = static_cast<int>(m_kseedsx[k]);
        int cy = static_cast<int>(m_kseedsy[k]);
        
        // Find minimum edge position in 3×3 neighborhood
        float min_edge = edgemag.at<float>(cy, cx);
        int best_x = cx;
        int best_y = cy;
        
        for (int i = 0; i < 8; ++i) {
            int nx = cx + dx8[i];
            int ny = cy + dy8[i];
            
            // Check bounds
            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                float edge_val = edgemag.at<float>(ny, nx);
                if (edge_val < min_edge) {
                    min_edge = edge_val;
                    best_x = nx;
                    best_y = ny;
                }
            }
        }
        
        // Update seed position and values
        m_kseedsx[k] = static_cast<float>(best_x);
        m_kseedsy[k] = static_cast<float>(best_y);
        m_kseeds_gray[k] = static_cast<float>(m_image.at<uchar>(best_y, best_x));
        m_kseeds_tex[k] = static_cast<float>(m_texture.at<uchar>(best_y, best_x)); 
    }
}

void LTriDPSuperpixelSLIC::getSeeds()
{
    // This implementation is different from standard SLIC)
    // Place seeds on regular grid with spacing = region_size
    // Offset by region_size/2 to center seeds in grid cells
    
    int xoff = m_region_size / 2;
    int yoff = m_region_size / 2;
    
    // Clear existing seeds
    m_kseedsx.clear();
    m_kseedsy.clear();
    m_kseeds_gray.clear();
    m_kseeds_tex.clear(); 
    
    // Generate grid of seeds
    for (int y = 0; y < m_height; y += m_region_size) {
        int Y = y + yoff;
        if (Y >= m_height) continue;
        
        for (int x = 0; x < m_width; x += m_region_size) {
            int X = x + xoff;
            if (X >= m_width) continue;
            
            // Add seed at grid position
            m_kseedsx.push_back(static_cast<float>(X));
            m_kseedsy.push_back(static_cast<float>(Y));
            m_kseeds_gray.push_back(static_cast<float>(m_image.at<uchar>(Y, X)));
            m_kseeds_tex.push_back(static_cast<float>(m_texture.at<uchar>(Y, X)));
        }
    }
}

void LTriDPSuperpixelSLIC::iterate(int num_iterations)
{
    if (num_iterations <= 0) {
        throw std::invalid_argument("Number of iterations must be positive");
    }
    
    performLTriDPSLIC(num_iterations);
}

void LTriDPSuperpixelSLIC::performLTriDPSLIC(int num_iterations)
{
    // Distance tracking matrix
    cv::Mat distvec(m_height, m_width, CV_32F);
    
    // Spatial distance weight
    // Standard SLIC: xywt = (S/m)²
    const float xywt = (static_cast<float>(m_region_size) / m_ruler) * 
                       (static_cast<float>(m_region_size) / m_ruler);
    
    // Main iteration loop
    for (int itr = 0; itr < num_iterations; itr++) {
        // Reset distance matrix to infinity
        distvec.setTo(std::numeric_limits<float>::max());
        
        // Step 1: Assign pixels to nearest cluster center
        for (int k = 0; k < m_numlabels; ++k) {
            // Define 2S × 2S search window around cluster center
            int cy = static_cast<int>(m_kseedsy[k]);
            int cx = static_cast<int>(m_kseedsx[k]);
            
            int y1 = std::max(0, cy - m_region_size);
            int y2 = std::min(m_height, cy + m_region_size);
            int x1 = std::max(0, cx - m_region_size);
            int x2 = std::min(m_width, cx + m_region_size);
            
            // Get cluster center values
            float center_gray = m_kseeds_gray[k];
            float center_tex = m_kseeds_tex[k];
            float center_x = m_kseedsx[k];
            float center_y = m_kseedsy[k];
            
            // Search neighborhood
            for (int y = y1; y < y2; ++y) {
                for (int x = x1; x < x2; ++x) {
                    // Get pixel values
                    float pixel_gray = static_cast<float>(m_image.at<uchar>(y, x));
                    float pixel_tex = static_cast<float>(m_texture.at<uchar>(y, x));  // ADDED: texture pixel value
                    
                    // Gray distance component
                    float dc = pixel_gray - center_gray;
                    dc = dc * dc;  // squared difference
                    
                    // Texture distance component 
                    float dt = pixel_tex - center_tex;                       // ADDED: texture distance
                    dt = dt * dt;  // squared difference                     // ADDED: texture distance
                    
                    // Spatial distance component
                    float dx_diff = static_cast<float>(x) - center_x;
                    float dy_diff = static_cast<float>(y) - center_y;
                    float ds = dx_diff * dx_diff + dy_diff * dy_diff;
                    
                    // Combined distance metric from paper:
                    // D = sqrt((dc/Nc)² + (dt/Nt)² + (ds/Ns)²)
                    // 
                    // Normalization constants:
                    // - Nc = max gray difference = 255 (for 8-bit images)
                    // - Nt = max texture difference = 255 (LTriDP is 0-255)
                    // - Ns = S (superpixel size)
                    //
                    // Simplify: D = sqrt(dc/255² + dt/255² + ds/S²)
                    //          = sqrt((dc + dt)/255² + ds/(S²))
                    //
                    // For computational efficiency, we use:
                    // D = (dc + dt) + ds/xywt
                    // where xywt = (S/m)² provides compactness control
                    
                    float dist = dc + dt + ds / xywt;
                    
                    // Assign to nearest cluster
                    if (dist < distvec.at<float>(y, x)) {
                        distvec.at<float>(y, x) = dist;
                        m_klabels.at<int>(y, x) = k;
                    }
                }
            }
        }
        
        // Step 2: Update cluster centers with gray-threshold filtering
        updateCenters();
    }
}

void LTriDPSuperpixelSLIC::updateCenters()
{
    // Calculate gray threshold α (standard deviation of image)
    const float gray_threshold = calculateGrayThreshold();
    
    // Accumulation arrays for each cluster
    std::vector<float> sigma_x(m_numlabels, 0.0f);
    std::vector<float> sigma_y(m_numlabels, 0.0f);
    std::vector<float> sigma_gray(m_numlabels, 0.0f);
    std::vector<float> sigma_tex(m_numlabels, 0.0f);                        // ADDED: texture accumulator
    std::vector<int> cluster_size(m_numlabels, 0);
    
    // Accumulate pixel values with gray-threshold filtering                // MODIFIED: gray filtering
    // (Paper modification #2)                                              // MODIFIED: new filtering logic
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            int label = m_klabels.at<int>(y, x);
            
            // Get pixel gray value
            float pixel_gray = static_cast<float>(m_image.at<uchar>(y, x));
            
            // Get cluster center gray value
            float center_gray = m_kseeds_gray[label];
            

            // Key modification: Only include pixels within gray threshold  // MODIFIED: conditional inclusion
            // |gray_center - gray_pixel| < α                               // MODIFIED: threshold comparison
            float gray_diff = std::abs(center_gray - pixel_gray);           // ADDED: compute gray difference
            
            if (gray_diff < gray_threshold) {                               // ADDED: threshold filter
                // Pixel passes threshold - include in center update
                sigma_x[label] += static_cast<float>(x);
                sigma_y[label] += static_cast<float>(y);
                sigma_gray[label] += pixel_gray;
                sigma_tex[label] += static_cast<float>(m_texture.at<uchar>(y, x));  // ADDED: accumulate texture
                cluster_size[label]++;
            }
        }
    }
    
    // Compute new cluster centers from filtered pixels
    for (int k = 0; k < m_numlabels; ++k) {
        if (cluster_size[k] > 0) {
            // Average filtered pixels to get new center
            float count = static_cast<float>(cluster_size[k]);
            m_kseedsx[k] = sigma_x[k] / count;
            m_kseedsy[k] = sigma_y[k] / count;
            m_kseeds_gray[k] = sigma_gray[k] / count;
            m_kseeds_tex[k] = sigma_tex[k] / count;                      // ADDED: update texture center
        }
    }
}

float LTriDPSuperpixelSLIC::calculateGrayThreshold() const            // ADDED: new function for threshold
{
    // Calculate standard deviation of image gray values
    // α = σ (standard deviation)
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(m_image, mean, stddev);
    
    return static_cast<float>(stddev[0]);
}

void LTriDPSuperpixelSLIC::getLabels(cv::Mat& labels_out) const
{
    labels_out = m_klabels.clone();
}

int LTriDPSuperpixelSLIC::getNumberOfSuperpixels() const
{
    return m_numlabels;
}

void LTriDPSuperpixelSLIC::getLabelContourMask(cv::Mat& mask, bool thick_line) const
{
    // Create binary mask highlighting superpixel boundaries
    
    int line_width = thick_line ? 2 : 1;
    
    mask = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar(0));
    
    // 8-connected neighborhood offsets
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    
    std::vector<bool> istaken(m_width * m_height, false);
    
    int mainindex = 0;
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            int current_label = m_klabels.at<int>(y, x);
            
            // Count neighbors with different labels
            int different_neighbors = 0;
            for (int i = 0; i < 8; ++i) {
                int nx = x + dx8[i];
                int ny = y + dy8[i];
                
                if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                    int neighbor_index = ny * m_width + nx;
                    if (!istaken[neighbor_index]) {
                        int neighbor_label = m_klabels.at<int>(ny, nx);
                        if (current_label != neighbor_label) {
                            different_neighbors++;
                        }
                    }
                }
            }
            
            // Mark as boundary if enough neighbors differ
            if (different_neighbors > line_width) {
                mask.at<uchar>(y, x) = 255;
                istaken[mainindex] = true;
            }
            mainindex++;
        }
    }
}

void LTriDPSuperpixelSLIC::enforceLabelConnectivity(int min_element_size)
{
    // Enforce connectivity by relabeling small disconnected components
    // Adapted from OpenCV SuperpixelSLIC::enforceLabelConnectivity()
    
    if (min_element_size == 0) return;
    if (min_element_size < 0 || min_element_size > 100) {
        throw std::invalid_argument("min_element_size must be in range [0, 100]");
    }
    
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};
    
    const int sz = m_width * m_height;
    const int supsz = sz / m_numlabels;
    
    // Calculate minimum superpixel size from percentage
    int div = static_cast<int>(100.0f / static_cast<float>(min_element_size) + 0.5f);
    int min_sp_sz = std::max(3, supsz / div);
    
    cv::Mat nlabels(m_height, m_width, CV_32S, cv::Scalar(std::numeric_limits<int>::max()));
    
    int label = 0;
    std::vector<int> xvec(sz);
    std::vector<int> yvec(sz);
    
    for (int y = 0; y < m_height; ++y) {
        for (int x = 0; x < m_width; ++x) {
            if (nlabels.at<int>(y, x) == std::numeric_limits<int>::max()) {
                nlabels.at<int>(y, x) = label;
                
                // Start flood fill for new segment
                xvec[0] = x;
                yvec[0] = y;
                
                // Find adjacent label for small segments
                int adjlabel = 0;
                for (int n = 0; n < 4; ++n) {
                    int nx = xvec[0] + dx4[n];
                    int ny = yvec[0] + dy4[n];
                    if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                        if (nlabels.at<int>(ny, nx) != std::numeric_limits<int>::max()) {
                            adjlabel = nlabels.at<int>(ny, nx);
                        }
                    }
                }
                
                int count = 1;
                for (int c = 0; c < count; ++c) {
                    for (int n = 0; n < 4; ++n) {
                        int nx = xvec[c] + dx4[n];
                        int ny = yvec[c] + dy4[n];
                        
                        if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                            if (nlabels.at<int>(ny, nx) == std::numeric_limits<int>::max() &&
                                m_klabels.at<int>(y, x) == m_klabels.at<int>(ny, nx)) {
                                xvec[count] = nx;
                                yvec[count] = ny;
                                nlabels.at<int>(ny, nx) = label;
                                count++;
                            }
                        }
                    }
                }
                
                // If segment too small, merge with adjacent
                if (count <= min_sp_sz) {
                    for (int c = 0; c < count; ++c) {
                        nlabels.at<int>(yvec[c], xvec[c]) = adjlabel;
                    }
                    label--;
                }
                label++;
            }
        }
    }
    
    // Update labels
    m_klabels = nlabels.clone();
    m_numlabels = label;
}

/*
 * Combine adjacent superpixels into super-duper-pixels if they're similar enough in color.
 * Uses average colors of superpixels to determine if they're similar enough in color.
 */
void LTriDPSuperpixelSLIC::duperizeWithAverage(const float max_distance, const bool use_duper_distance = false)
{
	// Graph of which superpixels are adjecent to each other
	// First dimension is each superpixel
	// Second dimension is index of each neighboring superpixel
	std::vector< std::set<int> > superpixel_neighbors;

	// Average colors of each superpixel
	// First dimension is each color channel
	// Second dimension is each superpixel
	std::vector< std::vector<float> > superpixel_average_colors;

	// The number of pixels in each superpixel
	std::vector<int> superpixel_population;

	this->findSuperpixelNeighborsAndAverages(superpixel_neighbors, superpixel_average_colors, superpixel_population);

	std::list<SuperDuperPixel> superduperpixels;
	std::vector<SuperDuperPixel*> superduperpixel_pointers;
	std::vector<std::list<SuperDuperPixel>::iterator> superduperpixel_iterators;

	this->groupSuperpixels
	(
		max_distance,
		use_duper_distance,
		superpixel_neighbors,
		superpixel_average_colors,
		superpixel_population,
		superduperpixels,
		superduperpixel_pointers,
		superduperpixel_iterators
	);

	// Stores which super-duper-pixel each superpixel belong to
	// super-duper-pixel value of -1 means it doesn't belong to a superduperpixel yet
	std::vector<int> superduperpixel_indexes(m_numlabels, -1);
	int superduperpixel_count = this->indexSuperduperpixels(superduperpixels, superduperpixel_indexes);

	this->assignSuperduperpixels(superduperpixel_indexes);

	m_numlabels = superduperpixel_count;
}

/*
 * Combine adjacent superpixels into super-duper-pixels if they're similar enough in color.
 * Uses (normalized) color histograms of superpixels to determine if they're similar enough in color.
 */
void LTriDPSuperpixelSLIC::duperizeWithHistogram(const int num_buckets[], const float distance, const bool use_duper_distance = false)
{
	// Graph of which superpixels are adjecent to each other
	// First dimension is each superpixel
	// Second dimension is index of each neighboring superpixel
	std::vector< std::set<int> > superpixel_neighbors(m_numlabels);

	// The number of pixels in each superpixel
	std::vector<int> superpixel_population(m_numlabels, 0);

	// Color histograms of each superpixel
	// First dimension is each color channel
	// Second dimension is each histogram basket
	// Third dimension is each superpixel
	std::vector< std::vector< std::vector<float> >> superpixel_color_histograms;

	this->findSuperpixelNeighborsAndHistograms
	(
		num_buckets,
		superpixel_neighbors,
		superpixel_color_histograms,
		superpixel_population
	);

	std::list<SuperDuperPixel> superduperpixels;
	std::vector<SuperDuperPixel*> superduperpixel_pointers;
	std::vector<std::list<SuperDuperPixel>::iterator> superduperpixel_iterators;

	this->groupSuperpixels
	(
		num_buckets,
		distance,
		use_duper_distance,
		superpixel_neighbors,
		superpixel_color_histograms,
		superpixel_population,
		superduperpixels,
		superduperpixel_pointers,
		superduperpixel_iterators
	);
	
	// Stores which super-duper-pixel each superpixel belong to
	// super-duper-pixel value of -1 means it doesn't belong to a superduperpixel yet
	std::vector<int> superduperpixel_indexes(m_numlabels, -1);
	int superduperpixel_count = this->indexSuperduperpixels(superduperpixels, superduperpixel_indexes);

	this->assignSuperduperpixels(superduperpixel_indexes);

	m_numlabels = superduperpixel_count;
}

void LTriDPSuperpixelSLIC::findSuperpixelNeighborsAndAverages
(
	std::vector< std::set<int> >& superpixel_neighbors,
	std::vector< std::vector<float> >& superpixel_average_colors,
	std::vector<int>& superpixel_population
)
{
	superpixel_neighbors = std::vector< std::set<int> >(m_numlabels);
	superpixel_average_colors = std::vector< std::vector<float> >(m_nr_channels, std::vector<float>(m_numlabels, 0));
	superpixel_population = std::vector<int>(m_numlabels, 0);
	// Loop through each pixel
	// Find superpixel connections
	// Get average color of superpixels
	for (int y = 0; y < m_height; y += 1)
	for (int x = 0; x < m_width; x += 1)
	{
		int current_superpixel = m_klabels.at<int>(y, x);
		this->linkNeighborSuperpixels(superpixel_neighbors, current_superpixel, x, y);
		// Keeps count of the number of pixels in each superpixel (for calculating average color)
		superpixel_population[current_superpixel] += 1;
		this->addColorsToAverages(superpixel_average_colors, current_superpixel, x, y);
	}
	
	// Loop through each superpixel
	// Remove superpixels being connected to themselves
	// Divide each superpixel average color value by the number of pixels in that superpixel to get the actual average
	for (int superpixel = 0; superpixel < m_numlabels; superpixel += 1)
	{
		superpixel_neighbors[superpixel].erase(superpixel);
		for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
		{
			superpixel_average_colors[color_channel][superpixel] /= superpixel_population[superpixel];
		}
	}
}

void LTriDPSuperpixelSLIC::findSuperpixelNeighborsAndHistograms
(
	const int num_buckets[],
	std::vector< std::set<int> >& superpixel_neighbors,
	std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
	std::vector<int>& superpixel_population
)
{
	superpixel_neighbors = std::vector< std::set<int> >(m_numlabels);
	superpixel_color_histograms = std::vector< std::vector< std::vector<float> >>(m_nr_channels);
	for (int channel = 0; channel < m_nr_channels; channel += 1)
	{
		superpixel_color_histograms[channel] = std::vector< std::vector<float> >(num_buckets[channel], std::vector<float>(m_numlabels, 0.0));
	}
	superpixel_population = std::vector<int>(m_numlabels, 0);

	// Loop through each pixel
	// Find superpixel connections
	// Get color histograms of superpixels
	for (int y = 0; y < m_height; y += 1)
	for (int x = 0; x < m_width; x += 1)
	{
		int current_superpixel = m_klabels.at<int>(y, x);
		this->linkNeighborSuperpixels(superpixel_neighbors, current_superpixel, x, y);
		// Keeps count of the number of pixels in each superpixel (for normalizing color histogram values to percentages)
		superpixel_population[current_superpixel] += 1;
		this->addColorsToHistograms(num_buckets, superpixel_color_histograms, current_superpixel, x, y);
	}

	// Loop through each superpixel
	// Remove superpixels being connected to themselves
	// Divide each superpixel color histogram value by the number of pixels in that superpixel to normalize them into percentages
	for (int superpixel = 0; superpixel < m_numlabels; superpixel += 1)
	{
		superpixel_neighbors[superpixel].erase(superpixel);
		for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
		for (int bucket = 0; bucket < num_buckets[color_channel]; bucket += 1)
		{
			superpixel_color_histograms[color_channel][bucket][superpixel] /= superpixel_population[superpixel];
		}
	}
}

void LTriDPSuperpixelSLIC::linkNeighborSuperpixels
(
	std::vector< std::set<int> >& superpixel_neighbors,
	const int current_superpixel,
	const int x,
	const int y
)
{
	// Create connections between adjacent superpixels based on where adjacent pixels are
	// in different superpixels
	bool not_left_column = x > 0;
	bool not_top_row = y > 0;
	if (not_left_column)
	{
		int superpixel_to_left = m_klabels.at<int>(y, x - 1);
		superpixel_neighbors[current_superpixel].insert(superpixel_to_left);
		superpixel_neighbors[superpixel_to_left].insert(current_superpixel);
	}
	if (not_top_row)
	{
		int superpixel_above = m_klabels.at<int>(y - 1, x);
		superpixel_neighbors[current_superpixel].insert(superpixel_above);
		superpixel_neighbors[superpixel_above].insert(current_superpixel);
	}
}

void LTriDPSuperpixelSLIC::addColorsToAverages
(
	std::vector< std::vector<float> >& superpixel_average_colors,
	const int current_superpixel,
	const int x,
	const int y
)
{
	// Get average colors and color histograms for each superpixel
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	{
		switch ( m_image.depth() )
		{
			case CV_8U:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<uchar>(y, x);
				break;

			case CV_8S:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<char>(y, x);
				break;

			case CV_16U:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<ushort>(y, x);
				break;

			case CV_16S:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<short>(y, x);
				break;

			case CV_32S:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<int>(y, x);
				break;

			case CV_32F:
				superpixel_average_colors[color_channel][current_superpixel] += m_image.at<float>(y, x);
				break;

			case CV_64F:
				superpixel_average_colors[color_channel][current_superpixel] += (float) m_image.at<double>(y, x);
				break;

			default:
				std::cerr << "Invalid matrix depth\n";
				break;
		}
	}
}

void LTriDPSuperpixelSLIC::addColorsToHistograms
(
	const int num_buckets[],
	std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
	const int current_superpixel,
	const int x,
	const int y
)
{
	// Get color histograms for each superpixel
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	{
		int bucket_index;
		switch ( m_image.depth() )
		{
			case CV_8U:
			{
				int max = std::numeric_limits<uchar>::max();
				int bucket_size = max / num_buckets[color_channel] + ((int) max % num_buckets[color_channel] != 0);
				bucket_index = m_image.at<uchar>(y, x) / bucket_size;
				break;
			}

			case CV_8S:
			{
				int max = std::numeric_limits<char>::max();
				int bucket_size = max / num_buckets[color_channel] + ((int) max % num_buckets[color_channel] != 0);
				bucket_index = m_image.at<char>(y, x) / bucket_size;
				break;
			}

			case CV_16U:
			{
				int max = std::numeric_limits<ushort>::max();
				int bucket_size = max / num_buckets[color_channel] + ((int) max % num_buckets[color_channel] != 0);
				bucket_index = m_image.at<ushort>(y, x) / bucket_size;
				break;
			}

			case CV_16S:
			{
				int max = std::numeric_limits<short>::max();
				int bucket_size = max / num_buckets[color_channel] + ((int) max % num_buckets[color_channel] != 0);
				bucket_index = m_image.at<short>(y, x) / bucket_size;
				break;
			}

			case CV_32S:
			{
				int max = std::numeric_limits<int>::max();
				int bucket_size = max / num_buckets[color_channel] + ((int) max % num_buckets[color_channel] != 0);
				bucket_index = m_image.at<int>(y, x) / bucket_size;
				break;
			}

			case CV_32F:
			{
				// Assume range lies between 0 and 1 for values that are float types
				bucket_index = ((int) (m_image.at<float>(y, x) * num_buckets[color_channel]));
				// Subtract 1 if the bucket index is too big (value of 1.0 * num_buckets would be out of bounds)
				bucket_index -= (int) (bucket_index == num_buckets[color_channel]);
				break;
			}

			case CV_64F:
			{
				// Assume range lies between 0 and 1 for values that are float types
				bucket_index = ((int) (m_image.at<double>(y, x) * num_buckets[color_channel]));
				// Subtract 1 if the bucket index is too big (value of 1.0 * num_buckets would be out of bounds)
				bucket_index -= (int) (bucket_index == num_buckets[color_channel]);
				break;
			}

			default:
				std::cerr << "Invalid matrix depth\n";
				break;
		}
		superpixel_color_histograms[color_channel][bucket_index][current_superpixel] += 1;
	}
}

// Groups superpixels into super-duper-pixels based on their average colors
void LTriDPSuperpixelSLIC::groupSuperpixels
(
	const float max_distance,
	const bool use_duper_distance,
	const std::vector< std::set<int> >& superpixel_neighbors,
	const std::vector< std::vector<float> >& superpixel_average_colors,
	const std::vector<int>& superpixel_population,
	std::list<SuperDuperPixel>& superduperpixels,
	std::vector<SuperDuperPixel*>& superduperpixel_pointers,
	std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators
)
{
	superduperpixel_pointers = std::vector<SuperDuperPixel*>(m_numlabels, NULL);
	superduperpixel_iterators = std::vector<std::list<SuperDuperPixel>::iterator>(m_numlabels, superduperpixels.end());
	// Loop through each superpixel
	// Group them together based on distances between average colors
	for (int superpixel = 0; superpixel < m_numlabels; superpixel += 1)
	{
		std::vector<float> average_colors(m_nr_channels);
		this->extractAverageColors(superpixel_average_colors, average_colors, superpixel);
		// Loop through each neighbor of this superpixel
		for (int neighbor: superpixel_neighbors[superpixel])
		{
			// Don't try to group together superpixels that are already grouped together
			if (superduperpixel_pointers[neighbor] == superduperpixel_pointers[superpixel] && superduperpixel_pointers[neighbor] != NULL)
			continue;

			// Get color distance to neighbor
			float neighbor_distance = this->getColorDistance
			(
				use_duper_distance,
				superduperpixels,
				superduperpixel_pointers,
				superpixel_average_colors,
				average_colors,
				superpixel,
				neighbor
			);

			// If the distance is close enough, group them into a super-duper-pixel
			if (neighbor_distance < max_distance)
			{
				this->combineIntoSuperDuperPixel
				(
					superduperpixels,
					superduperpixel_pointers,
					superduperpixel_iterators,
					superpixel_average_colors,
					average_colors,
					superpixel_population,
					superpixel,
					neighbor
				);
			}
		}

		if (superduperpixel_pointers[superpixel] == NULL)
		{
			superduperpixels.push_back(SuperDuperPixel(superpixel, average_colors, superpixel_population[superpixel]));
			superduperpixel_pointers[superpixel] = &superduperpixels.back();
			superduperpixel_iterators[superpixel] = --superduperpixels.end();
		}
	}
}

// Groups superpixels into super-duper-pixels based on their color histograms
void LTriDPSuperpixelSLIC::groupSuperpixels
(
	const int num_buckets[],
	const float max_distance,
	const bool use_duper_distance,
	const std::vector< std::set<int> >& superpixel_neighbors,
	const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
	const std::vector<int>& superpixel_population,
	std::list<SuperDuperPixel>& superduperpixels,
	std::vector<SuperDuperPixel*>& superduperpixel_pointers,
	std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators
)
{
	superduperpixel_pointers = std::vector<SuperDuperPixel*>(m_numlabels, NULL);
	superduperpixel_iterators = std::vector<std::list<SuperDuperPixel>::iterator>(m_numlabels, superduperpixels.end());
	// Loop through each superpixel
	// Group them together based on distances between average colors
	for (int superpixel = 0; superpixel < m_numlabels; superpixel += 1)
	{
		std::vector< std::vector<float> > color_histogram(m_nr_channels);
		this->extractColorHistogram(num_buckets, superpixel_color_histograms, color_histogram, superpixel);
		
		for (int neighbor: superpixel_neighbors[superpixel])
		{
			// Don't try to group together superpixels that are already grouped together
			if (superduperpixel_pointers[neighbor] == superduperpixel_pointers[superpixel] && superduperpixel_pointers[neighbor] != NULL)
			continue;

			// Get color distance to neighbor
			float neighbor_distance = this->getColorDistance
			(
				num_buckets,
				use_duper_distance,
				superduperpixels,
				superduperpixel_pointers,
				superpixel_color_histograms,
				color_histogram,
				superpixel,
				neighbor
			);

			// If the distance is close enough, group them into a super-duper-pixel
			if (neighbor_distance < max_distance)
			{
				this->combineIntoSuperDuperPixel
				(
					num_buckets,
					superduperpixels,
					superduperpixel_pointers,
					superduperpixel_iterators,
					superpixel_color_histograms,
					color_histogram,
					superpixel_population,
					superpixel,
					neighbor
				);
			}
		}

		if (superduperpixel_pointers[superpixel] == NULL)
		{
			superduperpixels.push_back(SuperDuperPixel(superpixel, color_histogram, superpixel_population[superpixel]));
			superduperpixel_pointers[superpixel] = &superduperpixels.back();
			superduperpixel_iterators[superpixel] = --superduperpixels.end();
		}
	}
}

// Gets the color distance between 2 superpixels' average colors
float LTriDPSuperpixelSLIC::getColorDistance
(
	const bool use_duper_distance,
	const std::list<SuperDuperPixel>& superduperpixels,
	const std::vector<SuperDuperPixel*>& superduperpixel_pointers,
	const std::vector< std::vector<float> >& superpixel_average_colors,
	const std::vector<float>& average_colors,
	const int superpixel,
	const int neighbor
)
{
	// If this superpixel is already in a superduperpixel, use the distance from that instead of the individual superpixel
	// Don't do this if use_duper_distance is false though
	std::vector<float> avg_colors = use_duper_distance && superduperpixel_pointers[superpixel] != NULL ?
	(*superduperpixel_pointers[superpixel]).get_average() :
	average_colors;

	// If the neighbor is already in a super-duper-pixel, use the distance to the whole super-duper-pixel it's in instead of just the neighbor
	// Don't do this if use_duper_distance is false though
	if (use_duper_distance && superduperpixel_pointers[neighbor] != NULL)
		return (*superduperpixel_pointers[neighbor]).distance_from(avg_colors);

	float neighbor_distance = 0;
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	{
		float difference = avg_colors[color_channel] - superpixel_average_colors[color_channel][neighbor];
		// opencv slic algorithm square diff before adding it to dist.
		// neighbor_distance += difference * difference;
		// Just take absolute value to do mahnattan distance instead.
		neighbor_distance += abs(difference);
	}
	// Just use manhattan distance here.
	// Could do this to be more precise (euclidian distance, would also need to square the diff above), but OpenCV
	// SLIC algorithm doesn't use it either.
	// neighbor_distance = sqrt(neighbor_distance);
	return neighbor_distance;
}

// Gets the color distance between 2 superpixels' color histograms
float LTriDPSuperpixelSLIC::getColorDistance
(
	const int num_buckets[],
	const bool use_duper_distance,
	const std::list<SuperDuperPixel>& superduperpixels,
	const std::vector<SuperDuperPixel*>& superduperpixel_pointers,
	const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms, 
	const std::vector< std::vector<float> >& color_histogram,
	const int superpixel,
	const int neighbor
)
{
	// If this superpixel is already in a superduperpixel, use the distance from that instead of the individual superpixel
	// Don't do this if use_duper_distance is false though
	std::vector< std::vector<float> > histogram = use_duper_distance && superduperpixel_pointers[superpixel] != NULL ?
	(*superduperpixel_pointers[superpixel]).get_histogram() :
	color_histogram;

	// If the neighbor is already in a super-duper-pixel, use the distance to the whole super-duper-pixel it's in instead of just the neighbor
	// Don't do this if use_duper_distance is false though
	if (use_duper_distance && superduperpixel_pointers[neighbor] != NULL)
		return (*superduperpixel_pointers[neighbor]).distance_from(color_histogram);

	float neighbor_distance = 0;
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	for (int bucket = 0; bucket < num_buckets[color_channel]; bucket += 1)
	{
		float difference = histogram[color_channel][bucket] - superpixel_color_histograms[color_channel][bucket][neighbor];
		// opencv slic algorithm square diff before adding it to dist.
		// neighbor_distance += difference * difference;
		// Just take absolute value to do mahnattan distance instead.
		neighbor_distance += abs(difference);
	}
	// Just use manhattan distance here.
	// Could do this to be more precise (euclidian distance, would also need to square the diff above), but OpenCV
	// SLIC algorithm doesn't use it either.
	// neighbor_distance = sqrt(neighbor_distance);
	return neighbor_distance;
}

// Gets a vector of the average colors for a superpixel
void LTriDPSuperpixelSLIC::extractAverageColors
(
	const std::vector< std::vector<float> >& superpixel_average_colors,
	std::vector<float>& average_colors,
	const int superpixel
)
{
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	{
		average_colors[color_channel] = superpixel_average_colors[color_channel][superpixel];
	}
}

// Gets a vector of the color histogram for a superpixel
void LTriDPSuperpixelSLIC::extractColorHistogram
(
	const int num_buckets[],
	const std::vector< std::vector< std::vector<float> >>& superpixel_color_histograms,
	std::vector< std::vector<float> >& color_histogram,
	const int superpixel
)
{
	for (int color_channel = 0; color_channel < m_nr_channels; color_channel += 1)
	{
		color_histogram[color_channel] = std::vector<float>(num_buckets[color_channel]);
		for (int bucket = 0; bucket < num_buckets[color_channel]; bucket += 1)
		{
			color_histogram[color_channel][bucket] = superpixel_color_histograms[color_channel][bucket][superpixel];
		}
	}
}

// Combines 2 superpixels into a super-duper-pixel using their average colors
void LTriDPSuperpixelSLIC::combineIntoSuperDuperPixel
(
	std::list<SuperDuperPixel>& superduperpixels,
	std::vector<SuperDuperPixel*>& superduperpixel_pointers,
	std::vector<std::list<SuperDuperPixel>::iterator>& superduperpixel_iterators,
	const std::vector< std::vector<float> >& superpixel_average_colors,
	const std::vector<float>& average_colors,
	const std::vector<int>& superpixel_population,
	const int superpixel,
	const int neighbor
)
{
	// If the neighbor is not already in a super-duper-pixel
	if (superduperpixel_pointers[neighbor] == NULL)
	{
		// If neither superpixels are in a super-duper-pixel
		if (superduperpixel_pointers[superpixel] == NULL)
		{
			// Create a new super-duper-pixel with the current superpixel
			superduperpixels.push_back(SuperDuperPixel(superpixel, average_colors, superpixel_population[superpixel]));
			superduperpixel_pointers[superpixel] = &superduperpixels.back();
			superduperpixel_iterators[superpixel] = --superduperpixels.end();
		}
		// Add the neighbor to the super-duper-pixel
		std::vector<float> neighbor_average_colors(m_nr_channels);
		this->extractAverageColors(superpixel_average_colors, neighbor_average_colors, neighbor);
		superduperpixel_pointers[superpixel]->add_superpixel(neighbor, neighbor_average_colors, superpixel_population[neighbor]);
		superduperpixel_pointers[neighbor] = superduperpixel_pointers[superpixel];
		superduperpixel_iterators[neighbor] = superduperpixel_iterators[superpixel];
	}
	// If the neighbor is already in a superpixel
	else
	{
		// If this superpixel is not in a super-duper-pixel yet
		if (superduperpixel_pointers[superpixel] == NULL)
		{
			// Add it to the neighbor's super-duper-pixel
			superduperpixel_pointers[neighbor]->add_superpixel(superpixel, average_colors, superpixel_population[superpixel]);
			superduperpixel_pointers[superpixel] = superduperpixel_pointers[neighbor];
			superduperpixel_iterators[superpixel] = superduperpixel_iterators[neighbor];
		}
		// If this superpixel is also already in a super-duper-pixel
		// And they're not in the same one
		else if (superduperpixel_pointers[superpixel] != superduperpixel_pointers[neighbor])
		{
			// Merge the superpixels (move all the superpixels from A to B and delete A)
			std::list<SuperDuperPixel>::iterator merging_superduperpixel = superduperpixel_iterators[neighbor];
			(*superduperpixel_pointers[superpixel]) += superduperpixel_pointers[neighbor];
			for (int connected_neighbor : superduperpixel_pointers[neighbor]->get_superpixels())
			{
				superduperpixel_pointers[connected_neighbor] = superduperpixel_pointers[superpixel];
				superduperpixel_iterators[connected_neighbor] = superduperpixel_iterators[superpixel];
			}
			superduperpixels.erase(merging_superduperpixel);
		}
	}
}

// Combines 2 superpixels into a super-duper-pixel using their color histograms
void LTriDPSuperpixelSLIC::combineIntoSuperDuperPixel
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
)
{
	// If the neighbor is not already in a super-duper-pixel
	if (superduperpixel_pointers[neighbor] == NULL)
	{
		// If neither superpixels are in a super-duper-pixel
		if (superduperpixel_pointers[superpixel] == NULL)
		{
			// Create a new super-duper-pixel with the current superpixel
			superduperpixels.push_back(SuperDuperPixel(superpixel, color_histogram, superpixel_population[superpixel]));
			superduperpixel_pointers[superpixel] = &superduperpixels.back();
			superduperpixel_iterators[superpixel] = --superduperpixels.end();
		}
		// Add the neighbor to the super-duper-pixel
		std::vector< std::vector<float>> neighbor_color_histogram(m_nr_channels);
		this->extractColorHistogram(num_buckets, superpixel_color_histograms, neighbor_color_histogram, neighbor);
		superduperpixel_pointers[superpixel]->add_superpixel(neighbor, neighbor_color_histogram, superpixel_population[neighbor]);
		superduperpixel_pointers[neighbor] = superduperpixel_pointers[superpixel];
		superduperpixel_iterators[neighbor] = superduperpixel_iterators[superpixel];
	}
	// If the neighbor is already in a superpixel
	else
	{
		// If this superpixel is not in a super-duper-pixel yet
		if (superduperpixel_pointers[superpixel] == NULL)
		{
			superduperpixel_pointers[neighbor]->add_superpixel(superpixel, color_histogram, superpixel_population[superpixel]);
			superduperpixel_pointers[superpixel] = superduperpixel_pointers[neighbor];
			superduperpixel_iterators[superpixel] = superduperpixel_iterators[neighbor];
		}
		// If this superpixel is also already in a super-duper-pixel
		// And they're not in the same one
		else if (superduperpixel_pointers[superpixel] != superduperpixel_pointers[neighbor])
		{
			// Merge the superpixels (move all the superpixels from A to B and delete A)
			std::list<SuperDuperPixel>::iterator merging_superduperpixel = superduperpixel_iterators[neighbor];
			(*superduperpixel_pointers[superpixel]) += superduperpixel_pointers[neighbor];
			for (int connected_neighbor : superduperpixel_pointers[neighbor]->get_superpixels())
			{
				superduperpixel_pointers[connected_neighbor] = superduperpixel_pointers[superpixel];
				superduperpixel_iterators[connected_neighbor] = superduperpixel_iterators[superpixel];
			}
			superduperpixels.erase(merging_superduperpixel);
		}
	}
}

// Gives super-duper-pixels indexes to assign to pixels as labels for what superpixel they're in
int LTriDPSuperpixelSLIC::indexSuperduperpixels
(
	const std::list<SuperDuperPixel>& superduperpixels,
	std::vector<int>& superduperpixel_indexes
)
{
	// Iterate through every super-duper-pixel and give them indexes starting at 0
	superduperpixel_indexes = std::vector<int>(m_numlabels, -1);
	int superduperpixel_count = 0;
	for (SuperDuperPixel sdp : superduperpixels)
	{
		for (int superpixel : sdp.get_superpixels())
		{
			superduperpixel_indexes[superpixel] = superduperpixel_count;
		}
		superduperpixel_count += 1;
	}
	return superduperpixel_count;
}

// Assigns new super-duper-pixel indexes to pixels in the image as labels for what superpixel they're in
void LTriDPSuperpixelSLIC::assignSuperduperpixels(const std::vector<int>& superduperpixel_indexes)
{
	// Change m_klabels so pixels use superduperpixel indexes instead of their old superpixel labels
	for (int y = 0; y < m_height; y += 1)
	for (int x = 0; x < m_width; x += 1)
	{
		m_klabels.at<int>(y, x) = superduperpixel_indexes[m_klabels.at<int>(y, x)];
	}
}

} // namespace ltridp
