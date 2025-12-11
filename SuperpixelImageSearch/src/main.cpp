#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cmath>
#include <array>
#include "json.hpp"
#include "sdp_slic.hpp"

namespace fs = std::filesystem;
using json   = nlohmann::json;

// PATH CONFIG
const std::string DATA_ROOT = "../../SuperpixelImageSearch/data";
const std::string INDEX_DIR = DATA_ROOT + "/coco2017/images/train2017";
const std::string QUERY_IMG = DATA_ROOT + "/coco2017/images/val2017/000000000139.jpg";
const std::string TRAIN_ANN = DATA_ROOT + "/coco2017/annotations/annotations/instances_train2017.json";
const std::string VAL_ANN   = DATA_ROOT + "/coco2017/annotations/annotations/instances_val2017.json";

// EXPERIMENT CONFIG
constexpr size_t MAX_IMAGES     = 2250;
constexpr bool   USE_ALL_IMAGES = true;
constexpr int    TOP_K          = 5;

// four cell sizes to test for SUPERPIXEL_SPATIAL
constexpr int SUPERPIXEL_SIZE_1 = 8;
constexpr int SUPERPIXEL_SIZE_2 = 32;
constexpr int SUPERPIXEL_SIZE_3 = 128;
constexpr int SUPERPIXEL_SIZE_4 = 512;

constexpr std::array<int, 4> SUPERPIXEL_SIZES = {
    SUPERPIXEL_SIZE_1,
    SUPERPIXEL_SIZE_2,
    SUPERPIXEL_SIZE_3,
    SUPERPIXEL_SIZE_4
};

constexpr int SUPERPIXEL_RESIZE_WIDTH  = 256;
constexpr int SUPERPIXEL_RESIZE_HEIGHT = 256;

// SD-SLIC custom pipeline parameters
constexpr int   SDSLIC_REGION_SIZE        = 100;
constexpr float SDSLIC_SMOOTHNESS         = 10.0f;
constexpr int   SDSLIC_MIN_SIZE_PERCENT   = 4;
constexpr int   SDSLIC_ITERATIONS         = 10;
constexpr float SDSLIC_HIST_DISTANCE      = 2.0f;
constexpr int   SDSLIC_HIST_BUCKETS[3]    = {8, 64, 64};
constexpr int   CUSTOM_FIXED_REGIONS      = 64;

// BASIC TYPES

enum class FeatureType    { SIFT, ORB };
enum class DescriptorMode { GLOBAL, SUPERPIXEL_SPATIAL, CUSTOM };

std::string featureTypeToString(FeatureType t) {
    return (t == FeatureType::SIFT) ? "SIFT" : "ORB";
}

std::string descriptorModeToString(DescriptorMode m) {
    switch (m) {
        case DescriptorMode::GLOBAL:             return "GLOBAL";
        case DescriptorMode::SUPERPIXEL_SPATIAL: return "SUPERPIXEL_SPATIAL";
        case DescriptorMode::CUSTOM:             return "CUSTOM";
        default:                                 return "UNKNOWN";
    }
}

// DATA INDEXING FOR COCO LABELS

struct COCOLabelIndex {
    std::unordered_map<std::string, std::unordered_set<int>> imageToCats;
    std::unordered_map<int, std::string> catIdToName;
};

void loadCOCOAnnotations(const std::string& annPath, COCOLabelIndex& index) {
    std::ifstream f(annPath);
    if (!f.is_open()) {
        std::cerr << "Could not open COCO annotation file: " << annPath << "\n";
        return;
    }

    json j;
    f >> j;

    if (j.contains("categories")) {
        for (const auto& category : j["categories"]) {
            int id = category.value("id", -1);
            std::string name = category.value("name", "");
            if (id >= 0 && !name.empty())
                index.catIdToName[id] = name;
        }
    }

    std::unordered_map<int, std::string> imageIdToFile;
    if (j.contains("images")) {
        for (const auto& img : j["images"]) {
            int id = img.value("id", -1);
            std::string fname = img.value("file_name", "");
            if (id >= 0 && !fname.empty())
                imageIdToFile[id] = fname;
        }
    }

    if (j.contains("annotations")) {
        for (const auto& ann : j["annotations"]) {
            int imgId = ann.value("image_id", -1);
            int catId = ann.value("category_id", -1);
            if (imgId < 0 || catId < 0) continue;

            auto it = imageIdToFile.find(imgId);
            if (it == imageIdToFile.end()) continue;

            index.imageToCats[it->second].insert(catId);
        }
    }

    std::cout << "Loaded COCO annotations from: " << annPath << "\n";
}

std::vector<int> getCategoriesForImage(const COCOLabelIndex& index,
                                       const std::string& fullPath) {
    std::string fname = fs::path(fullPath).filename().string();
    auto it = index.imageToCats.find(fname);
    if (it == index.imageToCats.end()) return {};
    return std::vector<int>(it->second.begin(), it->second.end());
}

std::string catIdsToString(const std::vector<int>& ids,
                           const COCOLabelIndex& index) {
    std::vector<std::string> names;
    names.reserve(ids.size());
    for (int id : ids) {
        auto it = index.catIdToName.find(id);
        names.push_back(it != index.catIdToName.end() ? it->second : ("id_" + std::to_string(id)));
    }
    std::sort(names.begin(), names.end());
    std::string out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out += "|";
        out += names[i];
    }
    return out;
}

// GRID SUPERPIXEL GENERATION

void makeGridSuperpixels(const cv::Mat& bgr,
                         cv::Mat& labels,
                         int& numSuperpixels,
                         int cellSize = 32) {
    CV_Assert(bgr.type() == CV_8UC3);
    const int h = bgr.rows;
    const int w = bgr.cols;

    const int gridX = (w + cellSize - 1) / cellSize;
    const int gridY = (h + cellSize - 1) / cellSize;

    labels.create(h, w, CV_32S);
    for (int y = 0; y < h; ++y) {
        int gy = y / cellSize;
        int* lblRow = labels.ptr<int>(y);
        for (int x = 0; x < w; ++x) {
            int gx = x / cellSize;
            lblRow[x] = gy * gridX + gx;
        }
    }

    numSuperpixels = gridX * gridY;
}

// FEATURE EXTRACTION

void computeFeatures(const cv::Mat& gray,
                     FeatureType type,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::Mat& descriptors,
                     int& descDim) {
    if (type == FeatureType::SIFT) {
        auto sift = cv::SIFT::create();
        sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        descDim = 128;
    } else {
        auto orb = cv::ORB::create(1000);
        orb->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
        descDim = 32;
    }

    if (descriptors.empty()) {
        descriptors = cv::Mat(0, descDim, CV_32F);
    } else if (descriptors.type() != CV_32F) {
        descriptors.convertTo(descriptors, CV_32F);
    }
}

cv::Mat globalDescriptorMean(const cv::Mat& desc, int dim) {
    if (desc.empty())
        return cv::Mat::zeros(1, dim, CV_32F);

    cv::Mat mean;
    cv::reduce(desc, mean, 0, cv::REDUCE_AVG, CV_32F);
    return mean;
}

std::vector<std::vector<int>> assignKeypointsToSuperpixels(
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat& labels) {

    CV_Assert(labels.type() == CV_32S);
    int h = labels.rows, w = labels.cols;

    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);
    int numSp = static_cast<int>(maxVal) + 1;

    std::vector<std::vector<int>> spToIndices(numSp);
    for (int i = 0; i < (int)keypoints.size(); ++i) {
        int x = static_cast<int>(std::round(keypoints[i].pt.x));
        int y = static_cast<int>(std::round(keypoints[i].pt.y));
        if (x >= 0 && x < w && y >= 0 && y < h) {
            int sp = labels.at<int>(y, x);
            spToIndices[sp].push_back(i);
        }
    }
    return spToIndices;
}

// GLOBAL DESCRIPTOR

cv::Mat buildGlobalDescriptor(const cv::Mat& bgr, FeatureType type) {
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kpts;
    cv::Mat desc;
    int dim = 0;
    computeFeatures(gray, type, kpts, desc, dim);

    cv::Mat globalFeat = globalDescriptorMean(desc, dim);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    cv::Scalar labMeanScalar = cv::mean(lab);

    cv::Mat labMean(1, 3, CV_32F);
    labMean.at<float>(0, 0) = (float)labMeanScalar[0];
    labMean.at<float>(0, 1) = (float)labMeanScalar[1];
    labMean.at<float>(0, 2) = (float)labMeanScalar[2];

    cv::Mat descriptor(1, dim + 3, CV_32F);
    globalFeat.copyTo(descriptor.colRange(0, dim));
    labMean.copyTo(descriptor.colRange(dim, dim + 3));

    cv::normalize(descriptor, descriptor);
    return descriptor;
}

// REGION DESCRIPTOR

cv::Mat buildRegionDescriptor(const cv::Mat& bgr,
                              const cv::Mat& labels,
                              int numRegions,
                              FeatureType type,
                              int fixedNumRegions = -1) {
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(labels.type() == CV_32S);
    CV_Assert(bgr.size() == labels.size());

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat desc;
    int descDim = 0;
    computeFeatures(gray, type, keypoints, desc, descDim);

    cv::Mat globalFeat = globalDescriptorMean(desc, descDim);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    cv::Scalar labMeanScalar = cv::mean(lab);
    cv::Mat labMean(1, 3, CV_32F);
    labMean.at<float>(0, 0) = (float)labMeanScalar[0];
    labMean.at<float>(0, 1) = (float)labMeanScalar[1];
    labMean.at<float>(0, 2) = (float)labMeanScalar[2];

    auto spToIdx = assignKeypointsToSuperpixels(keypoints, labels);

    int finalRegions = (fixedNumRegions > 0 ? fixedNumRegions : numRegions);
    cv::Mat regionMeans = cv::Mat::zeros(finalRegions, descDim, CV_32F);

    int usableRegions = std::min(numRegions, finalRegions);
    for (int sp = 0; sp < usableRegions; ++sp) {
        const auto& idxs = spToIdx[sp];
        if (idxs.empty() || desc.empty()) continue;

        cv::Mat sum = cv::Mat::zeros(1, descDim, CV_32F);
        for (int idx : idxs) sum += desc.row(idx);
        sum /= (float)idxs.size();
        sum.copyTo(regionMeans.row(sp));
    }

    cv::Mat regionFeat = regionMeans.reshape(1, 1);

    int totalDim = globalFeat.cols + 3 + regionFeat.cols;
    cv::Mat descriptor(1, totalDim, CV_32F);
    int off = 0;
    globalFeat.copyTo(descriptor.colRange(off, off + globalFeat.cols)); off += globalFeat.cols;
    labMean.copyTo(descriptor.colRange(off, off + 3));                 off += 3;
    regionFeat.copyTo(descriptor.colRange(off, off + regionFeat.cols));

    cv::normalize(descriptor, descriptor);
    return descriptor;
}

// SUPERPIXEL-SPATIAL DESCRIPTOR

cv::Mat buildSuperpixelDescriptor(const cv::Mat& bgr,
                                  FeatureType type,
                                  int cellSize) {
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(cellSize > 0);

    cv::Mat bgrSmall;
    cv::resize(bgr, bgrSmall,
               cv::Size(SUPERPIXEL_RESIZE_WIDTH, SUPERPIXEL_RESIZE_HEIGHT));

    cv::Mat labels;
    int numSp = 0;
    makeGridSuperpixels(bgrSmall, labels, numSp, cellSize);

    return buildRegionDescriptor(bgrSmall, labels, numSp, type, numSp);
}

// CUSTOM DESCRIPTOR (SD-SLIC, fixed 64 regions)

cv::Mat buildCustomDescriptor(const cv::Mat& bgr, FeatureType type) {
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    cv::Ptr<SuperpixelSLIC> slic =
        createSuperpixelSLIC(lab, SLIC, SDSLIC_REGION_SIZE, SDSLIC_SMOOTHNESS);

    slic->iterate(SDSLIC_ITERATIONS);
    slic->enforceLabelConnectivity(SDSLIC_MIN_SIZE_PERCENT);
    slic->duperizeWithHistogram(SDSLIC_HIST_BUCKETS, SDSLIC_HIST_DISTANCE);

    cv::Mat labels;
    slic->getLabels(labels);
    int numRegions = slic->getNumberOfSuperpixels();

    cv::Mat labelsClamped = labels;
    if (numRegions > CUSTOM_FIXED_REGIONS) {
        labelsClamped = cv::Mat(labels.size(), CV_32S);
        for (int y = 0; y < labels.rows; ++y) {
            const int* inRow  = labels.ptr<int>(y);
            int*       outRow = labelsClamped.ptr<int>(y);
            for (int x = 0; x < labels.cols; ++x) {
                int sp = inRow[x];
                if (sp >= CUSTOM_FIXED_REGIONS) sp = CUSTOM_FIXED_REGIONS - 1;
                outRow[x] = sp;
            }
        }
        numRegions = CUSTOM_FIXED_REGIONS;
    }

    return buildRegionDescriptor(bgr, labelsClamped, numRegions, type, CUSTOM_FIXED_REGIONS);
}

// VISUALS

cv::Mat visualizeGridSuperpixels(const cv::Mat& bgr, int cellSize) {
    CV_Assert(bgr.type() == CV_8UC3);
    CV_Assert(cellSize > 0);

    cv::Mat bgrSmall;
    cv::resize(bgr, bgrSmall,
               cv::Size(SUPERPIXEL_RESIZE_WIDTH, SUPERPIXEL_RESIZE_HEIGHT));

    cv::Mat labels;
    int numSp = 0;
    makeGridSuperpixels(bgrSmall, labels, numSp, cellSize);

    int h = bgrSmall.rows, w = bgrSmall.cols;

    cv::Mat meanBGR = cv::Mat::zeros(numSp, 3, CV_32F);
    std::vector<int> counts(numSp, 0);

    for (int y = 0; y < h; ++y) {
        const int* lblRow      = labels.ptr<int>(y);
        const cv::Vec3b* inRow = bgrSmall.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            int sp = lblRow[x];
            const cv::Vec3b& pix = inRow[x];
            meanBGR.at<float>(sp, 0) += pix[0];
            meanBGR.at<float>(sp, 1) += pix[1];
            meanBGR.at<float>(sp, 2) += pix[2];
            counts[sp]++;
        }
    }

    for (int sp = 0; sp < numSp; ++sp) {
        if (counts[sp] > 0) {
            meanBGR.at<float>(sp, 0) /= counts[sp];
            meanBGR.at<float>(sp, 1) /= counts[sp];
            meanBGR.at<float>(sp, 2) /= counts[sp];
        }
    }

    cv::Mat mosaic(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        const int* lblRow = labels.ptr<int>(y);
        cv::Vec3b* outRow = mosaic.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            int sp = lblRow[x];
            cv::Vec3b c;
            c[0] = static_cast<uchar>(std::clamp(meanBGR.at<float>(sp, 0), 0.0f, 255.0f));
            c[1] = static_cast<uchar>(std::clamp(meanBGR.at<float>(sp, 1), 0.0f, 255.0f));
            c[2] = static_cast<uchar>(std::clamp(meanBGR.at<float>(sp, 2), 0.0f, 255.0f));
            outRow[x] = c;
        }
    }

    cv::Mat mosaicFull;
    cv::resize(mosaic, mosaicFull, bgr.size(), 0, 0, cv::INTER_NEAREST);
    return mosaicFull;
}

cv::Mat visualizeSDSLICSuperpixels(const cv::Mat& bgr) {
    CV_Assert(bgr.type() == CV_8UC3);

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);

    cv::Ptr<SuperpixelSLIC> slic =
        createSuperpixelSLIC(lab, SLIC, SDSLIC_REGION_SIZE, SDSLIC_SMOOTHNESS);

    slic->iterate(SDSLIC_ITERATIONS);
    slic->enforceLabelConnectivity(SDSLIC_MIN_SIZE_PERCENT);
    slic->duperizeWithHistogram(SDSLIC_HIST_BUCKETS, SDSLIC_HIST_DISTANCE);

    cv::Mat contourMask;
    slic->getLabelContourMask(contourMask, true);

    cv::Mat vis = bgr.clone();
    vis.setTo(cv::Scalar(0, 0, 255), contourMask);
    return vis;
}

// DESCRIPTOR BUILDER

cv::Mat buildDescriptor(const cv::Mat& bgr,
                        FeatureType type,
                        DescriptorMode mode,
                        int superpixelCellSize) {
    switch (mode) {
    case DescriptorMode::GLOBAL:
        return buildGlobalDescriptor(bgr, type);
    case DescriptorMode::SUPERPIXEL_SPATIAL:
        return buildSuperpixelDescriptor(bgr, type, superpixelCellSize);
    case DescriptorMode::CUSTOM:
        return buildCustomDescriptor(bgr, type);
    }
    return buildGlobalDescriptor(bgr, type);
}

// IMAGE INDEX

struct ImageIndex {
    std::vector<std::string> filenames;
    cv::Mat features;

    void add(const std::string& fname, const cv::Mat& desc) {
        if (features.empty()) {
            features = desc.clone();
        } else {
            cv::vconcat(features, desc, features);
        }
        filenames.push_back(fname);
    }

    std::vector<std::pair<int, float>> search(const cv::Mat& query, int k) const {
        std::vector<std::pair<int, float>> res;
        if (features.empty()) return res;

        CV_Assert(query.rows == 1);
        cv::Mat qRepeat;
        cv::repeat(query, features.rows, 1, qRepeat);

        cv::Mat diff, dists;
        cv::pow(features - qRepeat, 2, diff);
        cv::reduce(diff, dists, 1, cv::REDUCE_SUM);
        cv::sqrt(dists, dists);

        res.reserve(features.rows);
        for (int i = 0; i < features.rows; ++i)
            res.emplace_back(i, dists.at<float>(i, 0));

        int kk = std::min(k, (int)res.size());
        std::partial_sort(res.begin(), res.begin() + kk, res.end(),
            [](auto& a, auto& b){ return a.second < b.second; });
        if ((int)res.size() > kk) res.resize(kk);
        return res;
    }
};

// OTHER UTILITIES

bool isImageFile(const fs::path& p) {
    if (!p.has_extension()) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

struct ImageJobResult {
    std::string path;
    cv::Mat desc;
    bool ok = false;
};

ImageJobResult processImageJob(const std::string& path,
                               FeatureType type,
                               DescriptorMode mode,
                               int superpixelCellSize) {
    ImageJobResult r;
    r.path = path;

    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not read " << path << std::endl;
        return r;
    }

    try {
        r.desc = buildDescriptor(img, type, mode, superpixelCellSize);
        r.ok   = true;
    } catch (const std::exception& e) {
        std::cerr << "Error processing " << path << ": " << e.what() << std::endl;
    }
    return r;
}

// CONFIGURATION AND STATS

struct ExperimentConfig {
    FeatureType    feature;
    DescriptorMode mode;
    int            superpixelCellSize;
    std::string    name;
};

struct ExperimentStats {
    std::string methodName;
    std::string featureName;
    std::string modeName;
    int         superpixelCellSize = 0;
    int         gridX = 0;
    int         gridY = 0;
    size_t      numIndexed = 0;
    double      indexTimeMs = 0.0;
    double      queryTimeMs = 0.0;
    double      precisionAtK = 0.0;
    std::string maxStr;
};

// EXPERIMENT RUNNER

ExperimentStats runExperiment(const ExperimentConfig& cfg,
                              const std::vector<std::string>& imagePaths,
                              const COCOLabelIndex& cocoIndex,
                              const cv::Mat& queryImg,
                              const std::vector<int>& queryCats,
                              const std::string& queryCatStr,
                              const std::unordered_set<int>& queryCatSet,
                              std::ofstream& fout,
                              const std::string& maxStr) {

    ExperimentStats stats;
    stats.featureName        = featureTypeToString(cfg.feature);
    stats.modeName           = descriptorModeToString(cfg.mode);
    stats.superpixelCellSize = cfg.superpixelCellSize;
    stats.maxStr             = maxStr;
    stats.methodName         = cfg.name + "_" + maxStr;

    std::cout << "\nExperiment: "    << cfg.name  << "\n";
    std::cout << "Feature type: "  << stats.featureName << "\n";
    std::cout << "Descriptor:  "   << stats.modeName    << "\n";

    if (cfg.mode == DescriptorMode::SUPERPIXEL_SPATIAL)
        std::cout << "Superpixel cell size: " << cfg.superpixelCellSize << "\n";

    if (cfg.mode == DescriptorMode::SUPERPIXEL_SPATIAL) {
        stats.gridX = (SUPERPIXEL_RESIZE_WIDTH  + cfg.superpixelCellSize - 1) / cfg.superpixelCellSize;
        stats.gridY = (SUPERPIXEL_RESIZE_HEIGHT + cfg.superpixelCellSize - 1) / cfg.superpixelCellSize;
    }

    auto t0 = std::chrono::steady_clock::now();

    std::vector<ImageJobResult> results(imagePaths.size());
    std::atomic<size_t> nextIndex{0};

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (!numThreads) numThreads = 4;

    auto worker = [&]() {
        while (true) {
            size_t idx = nextIndex.fetch_add(1);
            if (idx >= imagePaths.size()) break;
            results[idx] = processImageJob(imagePaths[idx],
                                           cfg.feature,
                                           cfg.mode,
                                           cfg.superpixelCellSize);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (unsigned int i = 0; i < numThreads; ++i)
        threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    ImageIndex index;
    size_t count = 0;
    for (const auto& r : results) {
        if (!r.ok) continue;
        index.add(r.path, r.desc);
        if (++count % 50 == 0)
            std::cout << "Indexed " << count << " images...\n";
    }

    auto t1 = std::chrono::steady_clock::now();
    stats.indexTimeMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.numIndexed = index.filenames.size();

    std::cout << "Total indexed images: " << stats.numIndexed << "\n";
    if (stats.numIndexed == 0) {
        std::cerr << "No images indexed for " << cfg.name << "\n";
        return stats;
    }

    auto tq0 = std::chrono::steady_clock::now();
    cv::Mat queryDesc = buildDescriptor(queryImg, cfg.feature,
                                        cfg.mode, cfg.superpixelCellSize);
    auto matches = index.search(queryDesc, TOP_K);
    auto tq1 = std::chrono::steady_clock::now();
    stats.queryTimeMs =
        std::chrono::duration<double, std::milli>(tq1 - tq0).count();

    std::cout << "\nTop " << TOP_K << " matches (" << cfg.name << "):\n";
    for (auto& [idx, dist] : matches)
        std::cout << "  " << index.filenames[idx] << "  (dist=" << dist << ")\n";

    std::string outDir = "../SuperpixelImageSearch/output/" + cfg.name + "_" + maxStr;
    fs::create_directories(outDir);

    try {
        std::cout << "\nSaving top " << TOP_K << " matches to: " << outDir << "\n";
        int rank = 1;
        for (auto& [idx, dist] : matches) {
            cv::Mat img = cv::imread(index.filenames[idx], cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "Could not reload " << index.filenames[idx] << " for saving.\n";
                continue;
            }
            std::string outPath = outDir + "/match_" + std::to_string(rank) + ".jpg";
            if (!cv::imwrite(outPath, img))
                std::cerr << "Failed to write " << outPath << "\n";
            rank++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving results for " << cfg.name << ": " << e.what() << "\n";
    }

    if (cfg.mode == DescriptorMode::SUPERPIXEL_SPATIAL) {
        try {
            std::string origPath = outDir + "/query_original.jpg";
            if (!cv::imwrite(origPath, queryImg))
                std::cerr << "Failed to write " << origPath << "\n";

            cv::Mat spVis = visualizeGridSuperpixels(queryImg, cfg.superpixelCellSize);
            std::string spPath =
                outDir + "/query_superpixels_cell" +
                std::to_string(cfg.superpixelCellSize) + ".jpg";

            if (!cv::imwrite(spPath, spVis))
                std::cerr << "Failed to write " << spPath << "\n";
            else
                std::cout << "Saved query superpixels to: " << spPath << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving superpixel viz for " << cfg.name
                      << ": " << e.what() << "\n";
        }
    } else if (cfg.mode == DescriptorMode::CUSTOM) {
        try {
            std::string origPath = outDir + "/query_original.jpg";
            if (!cv::imwrite(origPath, queryImg))
                std::cerr << "Failed to write " << origPath << "\n";

            cv::Mat sdslicVis = visualizeSDSLICSuperpixels(queryImg);
            std::string sdPath = outDir + "/query_sdslic_hist.jpg";
            if (!cv::imwrite(sdPath, sdslicVis))
                std::cerr << "Failed to write " << sdPath << "\n";
            else
                std::cout << "Saved SD-SLIC query viz to: " << sdPath << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving SD-SLIC viz for " << cfg.name
                      << ": " << e.what() << "\n";
        }
    }

    std::string queryFname = fs::path(QUERY_IMG).filename().string();
    int correct = 0;
    int rank = 1;

    for (auto& [idx, dist] : matches) {
        const std::string& matchPath = index.filenames[idx];
        std::string matchFname = fs::path(matchPath).filename().string();

        auto matchCats = getCategoriesForImage(cocoIndex, matchPath);
        std::string matchCatStr = catIdsToString(matchCats, cocoIndex);

        bool shareLabel = false;
        for (int c : matchCats) {
            if (queryCatSet.count(c)) {
                shareLabel = true;
                break;
            }
        }
        if (shareLabel) correct++;

        fout << stats.methodName << ","
             << stats.featureName << ","
             << stats.modeName << ","
             << cfg.superpixelCellSize << ","
             << stats.gridX << ","
             << stats.gridY << ","
             << maxStr << ","
             << stats.numIndexed << ","
             << queryFname << ","
             << "\"" << queryCatStr << "\","
             << rank << ","
             << matchFname << ","
             << "\"" << matchCatStr << "\","
             << (shareLabel ? 1 : 0) << ","
             << dist << ","
             << stats.indexTimeMs << ","
             << stats.queryTimeMs << "\n";

        rank++;
    }

    stats.precisionAtK = (double)correct / (double)TOP_K;
    std::cout << "Precision@" << TOP_K << " (COCO category match, " << cfg.name
              << ") = " << stats.precisionAtK << "\n";

    return stats;
}

// MAIN

int main(int, char**) {
    try {
        std::cout << "Program started.\n";
        std::cout << "Index dir: " << INDEX_DIR << "\n";
        std::cout << "Query img: " << QUERY_IMG << "\n";

        size_t maxImages = USE_ALL_IMAGES
            ? std::numeric_limits<size_t>::max()
            : MAX_IMAGES;

        std::string maxStr = (USE_ALL_IMAGES || maxImages == std::numeric_limits<size_t>::max())
            ? "all"
            : std::to_string(maxImages);

        std::cout << "Max images: " << maxStr << "\n";

        COCOLabelIndex cocoIndex;
        std::cout << "Loading train annotations...\n";
        loadCOCOAnnotations(TRAIN_ANN, cocoIndex);
        std::cout << "Loading val annotations...\n";
        loadCOCOAnnotations(VAL_ANN,   cocoIndex);
        std::cout << "Finished loading annotations.\n";

        std::vector<std::string> imagePaths;
        for (const auto& entry : fs::directory_iterator(INDEX_DIR)) {
            if (!entry.is_regular_file()) continue;
            if (!isImageFile(entry.path())) continue;
            imagePaths.push_back(entry.path().string());
            if (imagePaths.size() >= maxImages) break;
        }

        std::cout << "Found " << imagePaths.size() << " images to index.\n";
        if (imagePaths.empty()) {
            std::cerr << "No images found in INDEX_DIR.\n";
            return 1;
        }

        cv::Mat queryImg = cv::imread(QUERY_IMG, cv::IMREAD_COLOR);
        if (queryImg.empty()) {
            std::cerr << "Could not read query image.\n";
            return 1;
        }

        // origin visualizations
        try {
            std::string originDir = "../SuperpixelImageSearch/output/origin/";
            fs::create_directories(originDir);

            std::string origPath = originDir + "query_original.jpg";
            if (!cv::imwrite(origPath, queryImg))
                std::cerr << "Failed to write " << origPath << "\n";

            for (int cell : SUPERPIXEL_SIZES) {
                cv::Mat sp = visualizeGridSuperpixels(queryImg, cell);
                std::string spPath = originDir + "query_grid_cell" +
                                     std::to_string(cell) + ".jpg";
                if (!cv::imwrite(spPath, sp))
                    std::cerr << "Failed to write " << spPath << "\n";
            }

            cv::Mat sdslicVis = visualizeSDSLICSuperpixels(queryImg);
            std::string sdPath = originDir + "query_sdslic_hist.jpg";
            if (!cv::imwrite(sdPath, sdslicVis))
                std::cerr << "Failed to write " << sdPath << "\n";

            std::cout << "Saved origin visualizations in: " << originDir << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error saving origin visualizations: " << e.what() << "\n";
        }

        auto queryCats = getCategoriesForImage(cocoIndex, QUERY_IMG);
        std::string queryCatStr = catIdsToString(queryCats, cocoIndex);
        std::unordered_set<int> queryCatSet(queryCats.begin(), queryCats.end());

        if (queryCats.empty())
            std::cerr << "Warning: query image has no COCO categories.\n";
        else
            std::cout << "Query COCO categories: " << queryCatStr << "\n";

        // CSV output
        std::string csvDir  = "../SuperpixelImageSearch/output/csv/";
        fs::create_directories(csvDir);
        std::string csvFile = csvDir + "master_results.csv";

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

        // experiment configs
        std::vector<ExperimentConfig> experiments;
        experiments.push_back({ FeatureType::SIFT, DescriptorMode::GLOBAL, 0, "SIFT_GLOBAL" });
        experiments.push_back({ FeatureType::ORB,  DescriptorMode::GLOBAL, 0, "ORB_GLOBAL" });

        for (int cell : SUPERPIXEL_SIZES) {
            experiments.push_back({
                FeatureType::SIFT,
                DescriptorMode::SUPERPIXEL_SPATIAL,
                cell,
                "SIFT_SUPERPIXEL_SPATIAL_" + std::to_string(cell)
            });
            experiments.push_back({
                FeatureType::ORB,
                DescriptorMode::SUPERPIXEL_SPATIAL,
                cell,
                "ORB_SUPERPIXEL_SPATIAL_" + std::to_string(cell)
            });
        }

        // NOTE: CUSTOM_SDSLIC_SIFT moved to pipeline_demo.cpp
        // experiments.push_back({
        //     FeatureType::SIFT,
        //     DescriptorMode::CUSTOM,
        //     0,
        //     "CUSTOM_SDSLIC_SIFT"
        // });

        std::vector<ExperimentStats> allStats;
        allStats.reserve(experiments.size());

        for (const auto& cfg : experiments) {
            auto stats = runExperiment(cfg,
                                      imagePaths,
                                      cocoIndex,
                                      queryImg,
                                      queryCats,
                                      queryCatStr,
                                      queryCatSet,
                                      fout,
                                      maxStr);
            allStats.push_back(stats);
        }

        fout.close();
        std::cout << "Master CSV saved to: " << csvFile << "\n";

        // CSV summary
        std::string summaryFile = csvDir + "summary_results.csv";
        std::ofstream fsumm(summaryFile);
        if (!fsumm.is_open()) {
            std::cerr << "Failed to write summary CSV: " << summaryFile << "\n";
        } else {
            fsumm << "method,feature,descriptor_mode,superpixel_cell_size,grid_x,grid_y,"
                  << "max_images,num_indexed,precision_at_k,"
                  << "index_time_ms,query_time_ms,"
                  << "index_time_per_image_ms,query_time_per_image_ms\n";

            for (const auto& s : allStats) {
                double idxPerImg = (s.numIndexed > 0)
                    ? (s.indexTimeMs / (double)s.numIndexed)
                    : 0.0;
                double qryPerImg = (s.numIndexed > 0)
                    ? (s.queryTimeMs / (double)s.numIndexed)
                    : 0.0;

                fsumm << s.methodName << ","
                      << s.featureName << ","
                      << s.modeName << ","
                      << s.superpixelCellSize << ","
                      << s.gridX << ","
                      << s.gridY << ","
                      << s.maxStr << ","
                      << s.numIndexed << ","
                      << s.precisionAtK << ","
                      << s.indexTimeMs << ","
                      << s.queryTimeMs << ","
                      << idxPerImg << ","
                      << qryPerImg << "\n";
            }

            fsumm.close();
            std::cout << "Summary CSV saved to: " << summaryFile << "\n";
        }

        std::cout << "Done.\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
