#pragma once

#ifndef CENTERFACE_H
#define CENTERFACE_H

#include "core/base/detection.h"
#include "infer/centerface_backend.hpp"

using namespace ObjectDetection;

class CenterFace : public Detection {

public:
	CenterFace();
	CenterFace(const string &config_file);
	~CenterFace();
	void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& heatmap, const shared_ptr<TRTInfer::Tensor>& wh,
		const shared_ptr<TRTInfer::Tensor>& offset, const shared_ptr<TRTInfer::Tensor>& landmark,
		int stride, float threshold, vector<ccutil::FaceBox>& bboxs);
	int EngineInference(const Mat& image, vector<ccutil::FaceBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::FaceBox>>* result);

private:
	int strides_;
	map<int, string> detect_labels_;

};

#endif