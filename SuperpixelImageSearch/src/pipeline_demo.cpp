// pipeline_demo.cpp
// Runs ONLY the CUSTOM_SDSLIC_SIFT experiment as a pipeline demo,
// reusing all the code from main.cpp without modifying any other files.

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <limits>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// Include the full original main.cpp, but rename its main()
// so we can define our own main in this file.
#define main superpixel_main
#include "main.cpp"
#undef main

int main(int, char**)
{
    try {
        std::cout << "==========================================\n";
        std::cout << "  PIPELINE DEMO: CUSTOM_SDSLIC_SIFT ONLY\n";
        std::cout << "==========================================\n\n";

        // --------------------------------------------------------------------
        // 1) Determine max images & label string (reuse same logic)
        // --------------------------------------------------------------------
        size_t maxImages = USE_ALL_IMAGES
            ? std::numeric_limits<size_t>::max()
            : MAX_IMAGES;

        std::string maxStr = (USE_ALL_IMAGES || maxImages == std::numeric_limits<size_t>::max())
            ? std::string("all")
            : std::to_string(maxImages);

        std::cout << "Index dir: " << INDEX_DIR << "\n";
        std::cout << "Query img: " << QUERY_IMG << "\n";
        std::cout << "Max images (pipeline demo): " << maxStr << "\n\n";

        // --------------------------------------------------------------------
        // 2) Load COCO annotations (train + val) into COCOLabelIndex
        // --------------------------------------------------------------------
        COCOLabelIndex cocoIndex;
        std::cout << "Loading train annotations...\n";
        loadCOCOAnnotations(TRAIN_ANN, cocoIndex);
        std::cout << "Loading val annotations...\n";
        loadCOCOAnnotations(VAL_ANN,   cocoIndex);
        std::cout << "Finished loading annotations.\n\n";

        // --------------------------------------------------------------------
        // 3) Collect image paths from INDEX_DIR (respect maxImages)
        // --------------------------------------------------------------------
        std::vector<std::string> imagePaths;
        for (const auto& entry : fs::directory_iterator(INDEX_DIR)) {
            if (!entry.is_regular_file()) continue;
            if (!isImageFile(entry.path())) continue;
            imagePaths.push_back(entry.path().string());
            if (!USE_ALL_IMAGES && imagePaths.size() >= maxImages) break;
        }

        std::cout << "Found " << imagePaths.size()
                  << " images to index (pipeline demo).\n";
        if (imagePaths.empty()) {
            std::cerr << "No images found in INDEX_DIR.\n";
            return 1;
        }

        // --------------------------------------------------------------------
        // 4) Load query image
        // --------------------------------------------------------------------
        cv::Mat queryImg = cv::imread(QUERY_IMG, cv::IMREAD_COLOR);
        if (queryImg.empty()) {
            std::cerr << "Could not read query image.\n";
            return 1;
        }

        // --------------------------------------------------------------------
        // 5) Origin visualizations for the pipeline demo
        //    (same style as main.cpp, but this is just for CUSTOM)
        // --------------------------------------------------------------------
        try {
            std::string originDir = "../SuperpixelImageSearch/output/pipeline_origin/";
            fs::create_directories(originDir);

            std::string origPath = originDir + std::string("query_original.jpg");
            if (!cv::imwrite(origPath, queryImg))
                std::cerr << "Failed to write " << origPath << "\n";

            for (int cell : SUPERPIXEL_SIZES) {
                cv::Mat gridVis = visualizeGridSuperpixels(queryImg, cell);
                std::string spPath =
                    originDir + "query_grid_cell" + std::to_string(cell) + ".jpg";
                if (!cv::imwrite(spPath, gridVis))
                    std::cerr << "Failed to write " << spPath << "\n";
            }

            cv::Mat sdslicVis = visualizeSDSLICSuperpixels(queryImg);
            std::string sdPath = originDir + "query_sdslic_hist.jpg";
            if (!cv::imwrite(sdPath, sdslicVis))
                std::cerr << "Failed to write " << sdPath << "\n";

            std::cout << "Saved origin visualizations in: " << originDir << "\n\n";
        }
        catch (const std::exception& e) {
            std::cerr << "Error saving origin visualizations: "
                      << e.what() << "\n\n";
        }

        // --------------------------------------------------------------------
        // 6) COCO labels for query image
        // --------------------------------------------------------------------
        auto queryCats = getCategoriesForImage(cocoIndex, QUERY_IMG);
        std::string queryCatStr = catIdsToString(queryCats, cocoIndex);
        std::unordered_set<int> queryCatSet(queryCats.begin(), queryCats.end());

        if (queryCats.empty())
            std::cerr << "Warning: query image has no COCO categories.\n";
        else
            std::cout << "Query COCO categories: " << queryCatStr << "\n\n";

        // --------------------------------------------------------------------
        // 7) CSV output specifically for this pipeline demo
        // --------------------------------------------------------------------
        std::string csvDir  = "../SuperpixelImageSearch/output/csv/";
        fs::create_directories(csvDir);
        std::string csvFile = csvDir + "pipeline_demo_results.csv";

        std::ofstream fout(csvFile);
        if (!fout.is_open()) {
            std::cerr << "Failed to write CSV: " << csvFile << "\n";
            return 1;
        }

        fout << "method,feature,descriptor_mode,superpixel_cell_size,grid_x,grid_y,"
             << "max_images,num_indexed,"
             << "query_filename,query_categories,"
             << "match_rank,match_filename,match_categories,shares_label,distance,"
             << "index_time_ms,query_time_ms\n";

        // --------------------------------------------------------------------
        // 8) CUSTOM pipeline config: CUSTOM_SDSLIC_SIFT ONLY
        // --------------------------------------------------------------------
        ExperimentConfig customCfg {
            FeatureType::SIFT,
            DescriptorMode::CUSTOM,
            0,
            "CUSTOM_SDSLIC_SIFT"
        };

        std::cout << "==========================================\n";
        std::cout << "Running CUSTOM_SDSLIC_SIFT pipeline...\n";
        std::cout << "==========================================\n\n";

        ExperimentStats stats = runExperiment(
            customCfg,
            imagePaths,
            cocoIndex,
            queryImg,
            queryCats,
            queryCatStr,
            queryCatSet,
            fout,
            maxStr
        );

        fout.close();

        std::cout << "\nPipeline demo finished.\n";
        std::cout << "Images written under ../SuperpixelImageSearch/output/\n";
        std::cout << "Pipeline CSV: " << csvFile << "\n";
        std::cout << "Precision@" << TOP_K << " = "
                  << stats.precisionAtK << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "pipeline_demo.cpp: Unhandled exception: "
                  << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "pipeline_demo.cpp: Unknown exception.\n";
        return 1;
    }
}
