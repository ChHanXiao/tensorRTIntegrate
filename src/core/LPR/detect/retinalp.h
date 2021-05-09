#pragma once

#ifndef RETINALP_H
#define RETINALP_H

#include "core/base/detection.h"
#include "infer/retinalp_backend.hpp"

using namespace ObjectDetection;

class RetinaLP : public Detection {

public:
	RetinaLP();
	RetinaLP(const string& config_file);
	~RetinaLP();
	void GenerateAnchors();
	void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& conf, const shared_ptr<TRTInfer::Tensor>& offset,
		const shared_ptr<TRTInfer::Tensor>& landmark, Mat anchors_matrix, float threshold, vector<ccutil::LPRBox>& bboxs);
	int EngineInference(const Mat& image, vector<ccutil::LPRBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::LPRBox>>* result);

private:
	Mat anchors_matrix_;
	int total_pix_feature_;
	vector<vector<float>> anchor_;
	vector<int> strides_;
	vector<int> num_anchors_;
	map<int, string> detect_labels_;

};

#endif