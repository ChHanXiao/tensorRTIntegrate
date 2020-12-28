#pragma once

#ifndef YOLOV5_H
#define YOLOV5_H

#include "infer/yolov5_detect_backend.hpp"
#include "detection.h"
#include "yaml-cpp/yaml.h"

using namespace ObjectDetection;

class YOLOv5 : public Detection {
public:
	YOLOv5(const string &config_file);
	~YOLOv5();
	vector<ccutil::BBox> EngineInference(const Mat& image);
	vector<vector<ccutil::BBox>> EngineInferenceOptim(const vector<Mat>& images);

private:
	vector<int> strides_;
	vector<int> num_anchors_;
	map<int, string> detect_labels_;
	vector<vector<vector<float>>> anchor_grid_ ;

	//vector<pair<float, float >> anchor_grid_8 = { {13.0, 17.0}, {31.0, 25.0}, {24.0, 51.0}, {61.0, 45.0} };
	//vector<pair<float, float >> anchor_grid_16 = { {48.0, 102.0}, {119.0, 96.0}, {97.0, 189.0}, {217.0, 184.0} };
	//vector<pair<float, float >> anchor_grid_32 = { {171.0, 384.0}, {324.0, 451.0}, {616.0, 618.0}, {800.0, 800.0} };
};

#endif