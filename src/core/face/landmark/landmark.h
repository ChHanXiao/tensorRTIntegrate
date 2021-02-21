#pragma once

#ifndef LANDMARK_H
#define LANDMARK_H

#include "core/base/trtmodel.h"

class Landmarker : public TrtModel {

public:

	Landmarker();
	Landmarker(const string &config_file);
	~Landmarker();
	void PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat& image, vector<cv::Point2f>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<cv::Point2f>>* result);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !LANDMARK_H
