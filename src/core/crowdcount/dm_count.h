#pragma once
#ifndef DM_COUNT_H
#define DM_COUNT_H

#include "core/base/trtmodel.h"

class DMCount : public TrtModel {
public:
	DMCount();
	DMCount(const string &config_file);
	~DMCount();
	void PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat &image, int* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<int>* result);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !DM_COUNT_H
