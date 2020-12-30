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
};

#endif