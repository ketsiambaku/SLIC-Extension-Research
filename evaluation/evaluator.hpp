/**
 * @file evaluator.hpp
 * @brief Declaration of superpixel segmentation quality metrics.
 * @author Ketsia Mbaku
 *
 * Declares the SuperpixelEvaluator class that implements Achanta et al.'s
 * Under-Segmentation Error (UE) and Boundary Recall (BR) metrics.
 */

#ifndef SUPERPIXEL_EVALUATOR_HPP
#define SUPERPIXEL_EVALUATOR_HPP

#include <opencv2/core.hpp>

/**
 * @class SuperpixelEvaluator
 * @brief Computes quality metrics for superpixel segmentation.
 *
 * Provides static methods to evaluate superpixel segmentation quality using
 * standard metrics from the literature.
 */
class SuperpixelEvaluator {
public:
    /**
     * @brief Compute under-segmentation error.
     *
     * Measures how much superpixels "bleed" across ground truth boundaries.
     * For each ground truth region, accumulates the area of all superpixels
     * that overlap it by more than the threshold fraction.
     *
     * Formula: U = (1/N) * [Σ_i Σ_{S_k | |S_k ∩ G_i| > B} |S_k| - N]
     * where B = overlapFractionThreshold * |S_k|
     *
     * Preconditions:
     * superpixelLabels and groundTruthLabels must have same dimensions
     * Both inputs must be CV_32S type
     * overlapFractionThreshold must be in [0.0, 1.0]
     *
     * Parameters:
     * @param superpixelLabels Predicted superpixel label map (CV_32S).
     * @param groundTruthLabels Ground truth segmentation (CV_32S).
     * @param overlapFractionThreshold Minimum overlap ratio (default 0.05 = 5%).
     * @return Under-segmentation error in [0, +inf). Lower is better.
     *
     * Returns normalized error value, clamped to non-negative.
     */
    static double computeUnderSegmentationError(const cv::Mat& superpixelLabels, const cv::Mat& groundTruthLabels, double overlapFractionThreshold = 0.05);

    /**
     * @brief Compute boundary recall metric.
     *
     * Measures the fraction of ground truth boundary pixels that are within
     * a specified tolerance distance of predicted superpixel boundaries.
     *
     * Preconditions:
     * Both inputs must be CV_32S type
     * boundaryToleranceInPixels must be non-negative
     * 
     * Parameters:
     * @param superpixelLabelImage Predicted superpixel labels (CV_32S).
     * @param groundTruthLabelImage Ground truth labels (CV_32S).
     * @param boundaryToleranceInPixels Maximum distance for matching (default 2).
     * Boundary recall in [0, 1]. Higher is better.
     * Returns 1.0 if no ground truth boundaries exist (perfect recall).
     */
    static double computeBoundaryRecall(
        const cv::Mat& superpixelLabelImage,
        const cv::Mat& groundTruthLabelImage,
        int boundaryToleranceInPixels = 2);

private:
    /**
     * @brief Extract binary boundary mask from label image.
     *
     * A pixel is marked as boundary if any of its 8-connected neighbors
     * has a different label value.
     * Preconditions:
     * labelImage must be CV_32S type
     * 
     * Parameters:
     * @param labelImage Segmentation label map (CV_32S).
     * returns Binary boundary mask
     */
    static cv::Mat computeLabelBoundaryMask(const cv::Mat& labelImage);
};

#endif // SUPERPIXEL_EVALUATOR_HPP
