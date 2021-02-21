#pragma once

#ifndef ARCFACE_H
#define ARCFACE_H

#include "core/base/trtmodel.h"

class ArcFace : public TrtModel {

public:

	ArcFace();
	ArcFace(const string &config_file);
	~ArcFace();
	void PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat &image, vector<float>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<float>>* result);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !ARCFACE_H