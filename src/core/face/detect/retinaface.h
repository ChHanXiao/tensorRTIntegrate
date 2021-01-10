#pragma once

#ifndef RETINAFACE_H
#define RETINAFACE_H

#include "core/base/detection.h"
#include "infer/retinaface_backend.hpp"

using namespace ObjectDetection;

class RetinaFace : public Detection {

public:
	RetinaFace();
	RetinaFace(const string& config_file);
	~RetinaFace();
	void GenerateAnchors();
	int EngineInference(const Mat& image, vector<ccutil::FaceBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::FaceBox>>* result);

private:
	Mat anchors_matrix_;
	int total_pix_feature_;
	vector<vector<float>> anchor_;
	vector<int> strides_;
	vector<int> num_anchors_;
	map<int, string> detect_labels_;

};

#endif