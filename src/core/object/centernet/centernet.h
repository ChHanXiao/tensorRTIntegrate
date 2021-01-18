#pragma once

#ifndef CENTERNET_H
#define CENTERNET_H

#include "core/base/detection.h"
#include "infer/centernet_backend.hpp"

using namespace ObjectDetection;

class CenterNet : public Detection {

public:
	CenterNet();
	CenterNet(const string &config_file);
	~CenterNet();
	void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& outHM, const shared_ptr<TRTInfer::Tensor>& outHMPool,
		const shared_ptr<TRTInfer::Tensor>& outWH, const shared_ptr<TRTInfer::Tensor>& outOffset,
		int stride, float threshold, vector<ccutil::BBox>& bboxs);
	int EngineInference(const Mat& image, vector<ccutil::BBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::BBox>>* result);

private:
	int strides_;
	map<int, string> detect_labels_;

};

#endif // !CENTERNET_H