#pragma once

#ifndef YOLOV5_H
#define YOLOV5_H

#include "infer/yolov5_detect_backend.hpp"
#include "detection.h"

using namespace ObjectDetection;

class YOLOv5 : public Detection {
public:
	YOLOv5(const string &config_file);
	~YOLOv5();
	vector<ccutil::BBox> EngineInference(const Mat& image);
	vector<vector<ccutil::BBox>> EngineInferenceOptim(const vector<Mat>& images);

private:
	int strides_[3] = { 8, 16, 32 };
	vector<pair<float, float >> anchor_grid_8 = { {10.000000, 13.000000}, {16.000000, 30.000000}, {33.000000, 23.000000} };
	vector<pair<float, float >> anchor_grid_16 = { {30.000000, 61.000000}, {62.000000, 45.000000}, {59.000000, 119.000000} };
	vector<pair<float, float >> anchor_grid_32 = { {116.000000, 90.000000}, {156.000000, 198.000000}, {373.000000, 326.000000} };
};

#endif