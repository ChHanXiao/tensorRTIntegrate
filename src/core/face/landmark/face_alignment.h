#pragma once

#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include "core/base/trtmodel.h"

class FaceAlignment : public TrtModel {

public:

	FaceAlignment();
	FaceAlignment(const string &config_file);
	~FaceAlignment();
	void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat& image, vector<cv::Point2f>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<cv::Point2f>>* result);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !FACE_ALIGNMENT_H
