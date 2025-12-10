/**
 * file: test_with_mri_samples.cpp
 * Test preprocessing pipeline with real MRI images
 * Processes MRI images from data/input/ directory, applies preprocessing,
 * and saves results to data/output/ with statistics.
 * 
 * Author: Ketsia Mbaku
 * 
 * Disclaimer:
 * This test program was created with assistance from AI model
 * model: COPILOT
 * After writing the initial version of this program, I used
 * Prompt: "Output user friendly update to stdout at all checkpoints in the pipeline and add test input/output validation."
 * 
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>
#include "preprocessing.hpp"

namespace fs = std::filesystem;
using namespace ltridp_slic_improved;

// Create side-by-side comparison image
cv::Mat createComparison(const cv::Mat& original, const cv::Mat& processed) {
    cv::Mat comparison;
    cv::hconcat(original, processed, comparison);
    cv::line(comparison, 
             cv::Point(original.cols, 0), 
             cv::Point(original.cols, original.rows),
             cv::Scalar(255), 2);
    
    return comparison;
}

int main(int argc, char** argv) {
    std::string inputDir = "../data/input";
    std::string outputDir = "../data/output";

    // create output directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    std::vector<fs::path> imageFiles;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
            imageFiles.push_back(entry.path());
    }
    
    if (imageFiles.empty()) {
        std::cout << "No images found in " << inputDir << std::endl;
        std::cout << "Please add MRI images from Whole Brain Atlas to " << inputDir << "/" << std::endl;
        std::cout << std::endl;
        std::cout << "Recommended dataset: Whole Brain Atlas (WBA)" << std::endl;
        std::cout << "  URL: http://www.med.harvard.edu/aanlib/home.html" << std::endl;
        return 0;
    }
    
    std::cout << "Found " << imageFiles.size() << " images in " << inputDir << std::endl;
    
    Preprocessor preprocessor;
    int successCount = 0;
    int failCount = 0;
    
    for (const auto& imagePath : imageFiles) {
        std::string filename = imagePath.filename().string();
        std::cout << "Processing: " << filename << std::endl;
        
        cv::Mat input = cv::imread(imagePath.string(), cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            std::cerr << "  Error: Could not load image" << std::endl;
            failCount++;
            continue;
        }
        
        cv::Mat output;
        if (!preprocessor.enhance(input, output, 0.5)) {
            std::cerr << "  Error: Preprocessing failed" << std::endl;
            failCount++;
            continue;
        }

        std::string stem = imagePath.stem().string();
        std::string ext = imagePath.extension().string();
        std::string outputPath = outputDir + "/" + stem + "_preprocessed" + ext;
        
        if (!cv::imwrite(outputPath, output)) {
            std::cerr << "  Error: Could not save preprocessed image" << std::endl;
            failCount++;
            continue;
        }
                
        cv::Mat comparison = createComparison(input, output);
        std::string comparisonPath = outputDir + "/" + stem + "_comparison" + ext;
        cv::imwrite(comparisonPath, comparison);
        std::cout << std::endl;
        successCount++;
    }
    
    std::cout << "  Successfully processed: " << successCount << " images" << std::endl;
    if (failCount > 0) {
        std::cout << "  Failed: " << failCount << " images" << std::endl;
    }
    std::cout << "  Output directory: " << outputDir << std::endl;    
    return (failCount > 0) ? 1 : 0;
}
