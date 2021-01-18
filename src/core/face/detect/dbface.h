#pragma once

#ifndef DBFACE_H
#define DBFACE_H

#include "core/base/detection.h"
#include "infer/dbface_backend.hpp"

using namespace ObjectDetection;

class DBFace : public Detection {

public:
	DBFace();
	DBFace(const string &config_file);
	~DBFace();
	void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& outHM, const shared_ptr<TRTInfer::Tensor>& outHMPool,
		const shared_ptr<TRTInfer::Tensor>& outTLRB, const shared_ptr<TRTInfer::Tensor>& outLandmark,
		int stride, float threshold, vector<ccutil::FaceBox>& bboxs);
	int EngineInference(const Mat& image, vector<ccutil::FaceBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::FaceBox>>* result);

private:
	int strides_;
	map<int, string> detect_labels_;

};

#endif