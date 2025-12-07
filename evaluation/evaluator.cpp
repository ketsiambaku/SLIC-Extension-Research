/**
 * @file evaluator.cpp
 * @brief Implementation of superpixel segmentation quality metrics.
 * @author Ketsia Mbaku
 *
 * Implements the SuperpixelEvaluator class.
 */

#include "evaluator.hpp"
#include <opencv2/imgproc.hpp>
#include <unordered_map>

double SuperpixelEvaluator::computeUnderSegmentationError( const cv::Mat& superpixelLabels, const cv::Mat& groundTruthLabels, double overlapFractionThreshold) {
    CV_Assert(superpixelLabels.size() == groundTruthLabels.size());
    CV_Assert(superpixelLabels.type() == CV_32S &&
              groundTruthLabels.type() == CV_32S);
    CV_Assert(overlapFractionThreshold >= 0.0 &&
              overlapFractionThreshold <= 1.0);

    const int numRows    = superpixelLabels.rows;
    const int numCols    = superpixelLabels.cols;
    const int numPixels  = numRows * numCols;

    std::unordered_map<int, int> superpixelSizes;
    std::unordered_map<int, std::unordered_map<int, int>> gtToSuperpixelOverlap;

    for (int row = 0; row < numRows; row++) {
        const int* spRow = superpixelLabels.ptr<int>(row);
        const int* gtRow = groundTruthLabels.ptr<int>(row);

        for (int col = 0; col < numCols; ++col) {
            int superpixelId = spRow[col];
            int gtRegionId   = gtRow[col];

            superpixelSizes[superpixelId]++;                 
            gtToSuperpixelOverlap[gtRegionId][superpixelId]++;
        }
    }

    double summedAreaOverGt = 0.0;

    // Sum_i  Sum_{[S_k | |S_k ∩ G_i| > B]} |S_k|
    for (const auto& gtEntry : gtToSuperpixelOverlap) {
        const auto& superpixelOverlapMap = gtEntry.second;

        for (const auto& spEntry : superpixelOverlapMap) {
            int superpixelId          = spEntry.first;
            int intersectionSize      = spEntry.second;       
            int superpixelPixelCount  = superpixelSizes[superpixelId]; 

            double B = overlapFractionThreshold *
                       static_cast<double>(superpixelPixelCount);

            if (intersectionSize > B) {
                summedAreaOverGt += static_cast<double>(superpixelPixelCount);
            }
        }
    }

    // U = (1/N) * (summedAreaOverGt - N)
    double undersegmentationError =
        (summedAreaOverGt - static_cast<double>(numPixels)) /
        static_cast<double>(numPixels);

    if (undersegmentationError < 0.0)
        undersegmentationError = 0.0;
    return undersegmentationError;
}

cv::Mat SuperpixelEvaluator::computeLabelBoundaryMask(const cv::Mat& labelImage) {
    CV_Assert(labelImage.type() == CV_32S);
    const int numRows = labelImage.rows;
    const int numCols = labelImage.cols;
    cv::Mat boundaryMask = cv::Mat::zeros(numRows, numCols, CV_8U);

    // Check all 8 neighbors
    const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int row = 0; row < numRows; ++row) {
        const int* labelRowPtr = labelImage.ptr<int>(row);
        uchar* boundaryRowPtr = boundaryMask.ptr<uchar>(row);

        for (int col = 0; col < numCols; ++col) {
            const int currentLabel = labelRowPtr[col];
            bool pixelIsBoundary   = false;

            // Check all 8 neighbors
            for (int k = 0; k < 8; ++k) {
                const int neighborRow = row + dy[k];
                const int neighborCol = col + dx[k];

                // Boundary check
                if (neighborRow < 0 || neighborRow >= numRows ||
                    neighborCol < 0 || neighborCol >= numCols) {
                    continue;
                }

                // Different label = boundary
                if (labelImage.at<int>(neighborRow, neighborCol) != currentLabel) {
                    pixelIsBoundary = true;
                    break;
                }
            }

            if (pixelIsBoundary) {
                boundaryRowPtr[col] = 255;
            }
        }
    }

    return boundaryMask;
}

double SuperpixelEvaluator::computeBoundaryRecall(const cv::Mat& superpixelLabelImage, const cv::Mat& groundTruthLabelImage, int boundaryToleranceInPixels) {
    CV_Assert(superpixelLabelImage.size() == groundTruthLabelImage.size());
    CV_Assert(superpixelLabelImage.type() == CV_32S);
    CV_Assert(groundTruthLabelImage.type() == CV_32S);

    // Extract boundary masks from label maps
    cv::Mat superpixelBoundaryMask = computeLabelBoundaryMask(superpixelLabelImage);
    cv::Mat gtBoundaryMask = computeLabelBoundaryMask(groundTruthLabelImage);
    cv::Mat superpixelNonBoundaryMask;
    cv::bitwise_not(superpixelBoundaryMask, superpixelNonBoundaryMask);
    cv::Mat distanceToSuperpixelBoundary;
    cv::distanceTransform(superpixelNonBoundaryMask, distanceToSuperpixelBoundary, cv::DIST_L2, cv::DIST_MASK_3);

    double matchedGtBoundaryPixelCount   = 0.0;
    double totalGtBoundaryPixelCount     = 0.0;
    const int numRows = gtBoundaryMask.rows;
    const int numCols = gtBoundaryMask.cols;

    for (int row = 0; row < numRows; ++row) {
        const uchar* gtBoundaryRowPtr    = gtBoundaryMask.ptr<uchar>(row);
        const float* distanceRowPtr      = distanceToSuperpixelBoundary.ptr<float>(row);

        for (int col = 0; col < numCols; ++col) {
            if (gtBoundaryRowPtr[col] == 0)
                continue; // not a ground-truth boundary pixel
            totalGtBoundaryPixelCount += 1.0;

            if (distanceRowPtr[col] <= static_cast<float>(boundaryToleranceInPixels)) {
                matchedGtBoundaryPixelCount += 1.0;
            }
        }
    }
    if (totalGtBoundaryPixelCount == 0.0) {
        return 1.0; // recall is perfect
    }

    return matchedGtBoundaryPixelCount / totalGtBoundaryPixelCount;
}
