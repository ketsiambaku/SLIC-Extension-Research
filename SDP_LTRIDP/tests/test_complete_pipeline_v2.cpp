/**
 * test_complete_pipeline_v2.cpp
 * Test program for SDP-LTriDP superpixel segmentation with duperization
 * 
 * This program processes MRI images through the complete pipeline:
 * 1. Preprocessing (3D histogram + gamma enhancement)
 * 2. Feature Extraction (LTriDP texture descriptor)
 * 3. Superpixel Segmentation (LTriDP-enhanced SLIC)
 * 4. Super-Duper-Pixel creation via duperization
 * 
 * @author Ketsia Mbaku
*/

#include "preprocessing.hpp"
#include "feature_extraction.hpp"
#include "slic.hpp"
#include "../../evaluation/evaluator.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <vector>
#include <cmath>


namespace fs = std::filesystem;

struct ImageResult {
    std::string imageName;
    int regionSize;
    double opencvEdgeScore;
    double sdpEdgeScore;
    double duperEdgeScore;
    double opencvCompactness;
    double sdpCompactness;
    double duperCompactness;
};

cv::Mat createComparisonGrid(const cv::Mat& original, const cv::Mat& enhanced, const cv::Mat& features, 
                             const cv::Mat& opencv_slic_boundaries, const cv::Mat& boundaries_on_enhanced,
                             const cv::Mat& duperized_boundaries) {
    cv::Size target_size(original.cols, original.rows);
    
    cv::Mat enhanced_display, features_display;
    if (enhanced.size() != target_size) {
        cv::resize(enhanced, enhanced_display, target_size);
    } else {
        enhanced_display = enhanced.clone();
    }
    
    if (features.size() != target_size) {
        cv::resize(features, features_display, target_size);
    } else {
        features_display = features.clone();
    }
    
    // Convert grayscale images to BGR for red text labels
    cv::Mat original_bgr, enhanced_bgr, features_bgr;
    cv::cvtColor(original, original_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(enhanced_display, enhanced_bgr, cv::COLOR_GRAY2BGR);
    cv::cvtColor(features_display, features_bgr, cv::COLOR_GRAY2BGR);
    
    cv::putText(original_bgr, "1. Original", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    cv::putText(enhanced_bgr, "2. Enhanced", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    cv::putText(features_bgr, "3. Texture", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    cv::putText(opencv_slic_boundaries, "4. OpenCV", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    
    cv::Mat boundaries_bgr;
    cv::cvtColor(boundaries_on_enhanced, boundaries_bgr, cv::COLOR_GRAY2BGR);
    cv::putText(boundaries_bgr, "5. Improved", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    
    cv::Mat duperized_bgr;
    cv::cvtColor(duperized_boundaries, duperized_bgr, cv::COLOR_GRAY2BGR);
    cv::putText(duperized_bgr, "6. Duperize", cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    
    // Create 2×3 grid
    cv::Mat top_row, bottom_row, grid;
    cv::hconcat(original_bgr, enhanced_bgr, top_row);
    cv::hconcat(top_row, features_bgr, top_row);
    cv::hconcat(opencv_slic_boundaries, boundaries_bgr, bottom_row);
    cv::hconcat(bottom_row, duperized_bgr, bottom_row);
    cv::vconcat(top_row, bottom_row, grid);
    
    return grid;
}

bool processImage(const fs::path& input_path, const fs::path& output_dir, std::vector<ImageResult>& results) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Processing: " << input_path.filename() << "\n";
    std::cout << std::string(80, '=') << "\n";
    
    cv::Mat original = cv::imread(input_path.string(), cv::IMREAD_GRAYSCALE);
    if (original.empty()) {
        std::cerr << "Error: Could not load image: " << input_path << "\n";
        return false;
    }
    
    std::cout << "✓ Loaded image: " << original.cols << "×" << original.rows << " pixels\n";
    
    std::cout << "\nStep 1: Preprocessing...\n";
    ltridp_slic_improved::Preprocessor preprocessor;
    cv::Mat enhanced;
    preprocessor.enhance(original, enhanced, 0.5f);
    std::cout << "  ✓ Complete\n";
    
    std::cout << "\nStep 2: Feature Extraction (LTriDP)...\n";
    ltridp_slic_improved::FeatureExtractor feature_extractor;
    cv::Mat features;
    feature_extractor.extract(enhanced, features);
    std::cout << "  ✓ Complete\n";
    
    std::cout << "\nStep 3: Superpixel Segmentation (SDP-LTriDP SLIC)...\n";
    
    std::vector<int> region_sizes = {5, 10, 20, 30};
    
    for (int region_size : region_sizes) {
        std::cout << "\n  Region size: " << region_size << " pixels\n";

        // OpenCV SLIC baseline on original image
        cv::Mat original_clean = original.clone();
        cv::Ptr<cv::ximgproc::SuperpixelSLIC> opencv_slic = cv::ximgproc::createSuperpixelSLIC(
            original_clean, cv::ximgproc::SLIC, region_size, float(region_size));
        opencv_slic->iterate(10);
        opencv_slic->enforceLabelConnectivity(25);

        cv::Mat opencv_labels, opencv_boundaries;
        opencv_slic->getLabels(opencv_labels);
        opencv_slic->getLabelContourMask(opencv_boundaries, true);

        // Draw white boundaries on original (convert to BGR)
        cv::Mat opencv_result_bgr;
        cv::cvtColor(original_clean, opencv_result_bgr, cv::COLOR_GRAY2BGR);
        opencv_result_bgr.setTo(cv::Scalar(255, 255, 255), opencv_boundaries);

        int opencv_superpixels = opencv_slic->getNumberOfSuperpixels();
        std::cout << "    OpenCV SLIC superpixels: " << opencv_superpixels << "\n";

        // Compute edge alignment and compactness for OpenCV SLIC
        double opencv_edge_score = SuperpixelEvaluator::computeEdgeAlignmentScore(opencv_labels, enhanced, 2);
        double opencv_compactness = SuperpixelEvaluator::computeAverageCompactness(opencv_labels);

        // SDP-LTriDP SLIC on enhanced image with features
        const float compactness_ratio = 1.0f;
        float ruler = compactness_ratio * static_cast<float>(region_size);
        sdp_ltridp::SDPLTriDPSLIC slic(enhanced, features, region_size, ruler);

        slic.iterate(10);

        int num_superpixels = slic.getNumberOfSuperpixels();
        std::cout << "    Number of superpixels: " << num_superpixels << "\n";

        cv::Mat labels;
        slic.getLabels(labels);
        slic.enforceLabelConnectivity(25);      
        int final_superpixels = slic.getNumberOfSuperpixels();
        std::cout << "    After connectivity: " << final_superpixels << " superpixels\n";

        slic.getLabels(labels);
        cv::Mat boundaries;
        slic.getLabelContourMask(boundaries);

        // Save boundaries BEFORE duperization for "5. Improved" panel
        cv::Mat enhanced_with_boundaries = enhanced.clone();
        enhanced_with_boundaries.setTo(255, boundaries);

        // Compute edge alignment and compactness for SDP-LTriDP SLIC
        double sdp_edge_score = SuperpixelEvaluator::computeEdgeAlignmentScore(labels, enhanced, 2);
        double sdp_compactness = SuperpixelEvaluator::computeAverageCompactness(labels);

        // Step 4: Duperization
        std::cout << "    Duperizing with average...\n";
        float duperize_distance = 10.0f;  // Color distance threshold
        slic.duperizeWithAverage(duperize_distance, true);

        int num_superduperpixels = slic.getNumberOfSuperpixels();
        std::cout << "      After duperize: " << num_superduperpixels << " super-duper-pixels\n";

        // Get duperized boundaries (after duperization modifies labels)
        cv::Mat duperized_boundaries;
        slic.getLabelContourMask(duperized_boundaries);

        cv::Mat enhanced_with_duperized = enhanced.clone();
        enhanced_with_duperized.setTo(255, duperized_boundaries);

        // Compute edge alignment and compactness for duperized superpixels
        cv::Mat duperized_labels;
        slic.getLabels(duperized_labels);
        double duper_edge_score = SuperpixelEvaluator::computeEdgeAlignmentScore(duperized_labels, enhanced, 2);
        double duper_compactness = SuperpixelEvaluator::computeAverageCompactness(duperized_labels);

        // Create clean copies for comparison grid
        cv::Mat original_for_grid = original.clone();
        cv::Mat enhanced_for_grid = enhanced.clone();
        cv::Mat features_for_grid = features.clone();

        cv::Mat comparison_grid = createComparisonGrid(original_for_grid, enhanced_for_grid, features_for_grid, 
                                                       opencv_result_bgr, enhanced_with_boundaries,
                                                       enhanced_with_duperized);

        int boundary_pixels = cv::countNonZero(boundaries);
        float boundary_percentage = 100.0f * static_cast<float>(boundary_pixels) / 
                                   static_cast<float>(enhanced.rows * enhanced.cols);
        std::cout << "    Boundary pixels: " << boundary_pixels 
                  << " (" << std::fixed << std::setprecision(2) 
                  << boundary_percentage << "%)\n";

        std::string base_name = input_path.stem().string();
        std::string size_suffix = "_S" + std::to_string(region_size);

        fs::path boundaries_path = output_dir / (base_name + size_suffix + "_boundaries.png");
        fs::path duperized_path = output_dir / (base_name + size_suffix + "_duperized.png");
        fs::path grid_path = output_dir / (base_name + size_suffix + "_pipeline.png");

        cv::imwrite(boundaries_path.string(), enhanced_with_boundaries);
        cv::imwrite(duperized_path.string(), enhanced_with_duperized);
        cv::imwrite(grid_path.string(), comparison_grid);

        std::cout << "    ✓ Saved: " << boundaries_path.filename() << "\n";
        std::cout << "    ✓ Saved: " << duperized_path.filename() << "\n";
        std::cout << "    ✓ Saved: " << grid_path.filename() << "\n";

        // Store results for summary table
        ImageResult result;
        result.imageName = input_path.stem().string();
        result.regionSize = region_size;
        result.opencvEdgeScore = opencv_edge_score;
        result.sdpEdgeScore = sdp_edge_score;
        result.duperEdgeScore = duper_edge_score;
        result.opencvCompactness = opencv_compactness;
        result.sdpCompactness = sdp_compactness;
        result.duperCompactness = duper_compactness;
        results.push_back(result);
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   SDP-LTriDP Superpixel Segmentation - Pipeline Test V2            ║\n";
    std::cout << "║   With Super-Duper-Pixel Duperization                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════════╝\n\n";
    
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_directory> <output_directory>\n";
        std::cerr << "\n";
        std::cerr << "Example:\n";
        std::cerr << "  " << argv[0] << " ../../ltridp/data/input ../../ltridp/data/output_v2\n";
        std::cerr << "\n";
        return 1;
    }
    
    fs::path input_dir(argv[1]);
    fs::path output_dir(argv[2]);
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << input_dir << "\n";
        return 1;
    }
    
    fs::create_directories(output_dir);
    std::cout << "Input directory:  " << fs::absolute(input_dir) << "\n";
    std::cout << "Output directory: " << fs::absolute(output_dir) << "\n";
    
    std::vector<fs::path> image_files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
                ext == ".bmp" || ext == ".tif" || ext == ".tiff") {
                image_files.push_back(entry.path());
            }
        }
    }
    
    if (image_files.empty()) {
        std::cerr << "\nError: No image files found in " << input_dir << "\n";
        std::cerr << "Supported formats: .png, .jpg, .jpeg, .bmp, .tif, .tiff\n";
        return 1;
    }
    
    std::cout << "\nFound " << image_files.size() << " image(s) to process\n";
    
    int success_count = 0;
    int failure_count = 0;
    std::vector<ImageResult> results;

    for (const auto& image_path : image_files) {
        if (processImage(image_path, output_dir, results)) {
            success_count++;
        } else {
            failure_count++;
        }
    }

    // Summary
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Processing Complete\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Successfully processed: " << success_count << " image(s)\n";
    if (failure_count > 0) {
        std::cout << "Failed: " << failure_count << " image(s)\n";
    }

    // Display edge alignment and compactness table
    if (!results.empty()) {
        std::cout << "\n" << std::string(120, '=') << "\n";
        std::cout << "Edge Alignment & Compactness (Lower is More Compact)\n";
        std::cout << std::string(120, '=') << "\n";
        std::cout << std::left << std::setw(20) << "Image"
                  << std::setw(12) << "Region Size"
                  << std::setw(18) << "OpenCV SLIC (EA)"
                  << std::setw(18) << "SDP-LTriDP (EA)"
                  << std::setw(18) << "Duperized (EA)"
                  << std::setw(18) << "OpenCV SLIC (C)"
                  << std::setw(18) << "SDP-LTriDP (C)"
                  << std::setw(18) << "Duperized (C)"
                  << "EA Improv."
                  << "\n";
        std::cout << std::string(120, '-') << "\n";
        double total_opencv = 0.0, total_sdp = 0.0, total_duper = 0.0;
        double total_opencv_c = 0.0, total_sdp_c = 0.0, total_duper_c = 0.0;
        for (const auto& result : results) {
            double improvement = result.sdpEdgeScore - result.opencvEdgeScore;
            std::cout << std::left << std::setw(20) << result.imageName
                      << std::setw(12) << result.regionSize
                      << std::fixed << std::setprecision(4)
                      << std::setw(18) << result.opencvEdgeScore
                      << std::setw(18) << result.sdpEdgeScore
                      << std::setw(18) << result.duperEdgeScore
                      << std::setw(18) << result.opencvCompactness
                      << std::setw(18) << result.sdpCompactness
                      << std::setw(18) << result.duperCompactness
                      << (improvement >= 0 ? "+" : "") << improvement << "\n";
            total_opencv += result.opencvEdgeScore;
            total_sdp += result.sdpEdgeScore;
            total_duper += result.duperEdgeScore;
            total_opencv_c += result.opencvCompactness;
            total_sdp_c += result.sdpCompactness;
            total_duper_c += result.duperCompactness;
        }
        std::cout << std::string(120, '-') << "\n";
        double avg_opencv = total_opencv / results.size();
        double avg_sdp = total_sdp / results.size();
        double avg_duper = total_duper / results.size();
        double avg_opencv_c = total_opencv_c / results.size();
        double avg_sdp_c = total_sdp_c / results.size();
        double avg_duper_c = total_duper_c / results.size();
        double avg_improvement = avg_sdp - avg_opencv;
        std::cout << std::left << std::setw(20) << "AVERAGE"
                  << std::setw(12) << ""
                  << std::fixed << std::setprecision(4)
                  << std::setw(18) << avg_opencv
                  << std::setw(18) << avg_sdp
                  << std::setw(18) << avg_duper
                  << std::setw(18) << avg_opencv_c
                  << std::setw(18) << avg_sdp_c
                  << std::setw(18) << avg_duper_c
                  << (avg_improvement >= 0 ? "+" : "") << avg_improvement << "\n";
        std::cout << std::string(120, '=') << "\n";
    }

    std::cout << "\nOutput files saved to: " << fs::absolute(output_dir) << "\n";
    std::cout << "\nGenerated files per image (for each region size S=5,10,20,30):\n";
    std::cout << "  *_S{N}_boundaries.png    - SDP-LTriDP boundaries on enhanced image\n";
    std::cout << "  *_S{N}_duperized.png     - Super-duper-pixel boundaries\n";
    std::cout << "  *_S{N}_pipeline.png      - Complete pipeline comparison\n";
    std::cout << "\n";

    return (failure_count == 0) ? 0 : 1;
}
