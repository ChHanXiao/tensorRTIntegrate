#pragma once

#ifndef ARCFACE_H
#define ARCFACE_H

#include "trtmodel.h"
#include "yaml-cpp/yaml.h"

class ArcFace : public TrtModel {
public:
	ArcFace(const string &config_file);
	~ArcFace();
	void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	vector<float> EngineInference(const Mat &image);
public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !ARCFACE_H