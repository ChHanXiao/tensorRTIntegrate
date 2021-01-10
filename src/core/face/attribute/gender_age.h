#pragma once

#ifndef GENDER_AGE_H
#define GENDER_AGE_H

#include "core/base/trtmodel.h"

class GenderAge : public TrtModel {

public:
	GenderAge();
	GenderAge(const string &config_file);
	~GenderAge();
	void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat &image, ccutil::FaceAttribute* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<ccutil::FaceAttribute>* result);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !GENDER_AGE_H
