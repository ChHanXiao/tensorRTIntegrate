#pragma once

#ifndef CENTERFACE_H
#define CENTERFACE_H

#include "detection.h"
#include "yaml-cpp/yaml.h"
#include "infer/centerface_backend.hpp"

using namespace ObjectDetection;

class CenterFace : public Detection {
public:
	CenterFace(const string &config_file);
	~CenterFace();

	vector<ccutil::FaceBox> EngineInference(const Mat &image);
	vector<vector<ccutil::FaceBox>> EngineInferenceOptim(const vector<Mat> &images);
private:
	int strides_;
	map<int, string> detect_labels_;

};

#endif