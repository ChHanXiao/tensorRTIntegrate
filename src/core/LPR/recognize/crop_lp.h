#pragma once
#ifndef CROP_LP_H
#define CROP_LP_H

#include <cc_util.hpp>

int crop_lp(const cv::Mat& img_src, const ccutil::LPRBox lpbox, cv::Mat* img_crop);
const cv::Point2f points_ref[4] = { cv::Point2f{0.0, 0.0}, cv::Point2f{94.0, 0.0}, cv::Point2f{0.0, 24.0}, cv::Point2f{94.0, 24.0} };
const cv::Size plate_size = cv::Size(94, 24);

#endif // CROP_LP_H
