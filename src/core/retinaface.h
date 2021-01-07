#pragma once

#ifndef RETINAFACE_H
#define RETINAFACE_H

#include "detection.h"
#include "yaml-cpp/yaml.h"
#include "infer/retinaface_backend.hpp"


using namespace ObjectDetection;

class RetinaFace : public Detection {
public:
	RetinaFace(const string &config_file);
	~RetinaFace();
	void GenerateAnchors();
	vector<ccutil::FaceBox> EngineInference(const Mat& image);
	vector<vector<ccutil::FaceBox>> EngineInferenceOptim(const vector<Mat>& images);

private:
	Mat anchors_matrix_;
	int total_pix_feature_;
	vector<vector<float>> anchor_;
	vector<int> strides_;
	vector<int> num_anchors_;
	map<int, string> detect_labels_;

};

#endif