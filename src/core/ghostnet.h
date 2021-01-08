#pragma once

#ifndef GHOSTNET_H
#define GHOSTNET_H

#include "trtmodel.h"
#include "yaml-cpp/yaml.h"

class GhostNet : public TrtModel {
public:
	GhostNet(const string &config_file);
	~GhostNet();
	void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat &image);
	vector<int> EngineInferenceOptim(const vector<Mat>& images);
public:
	string labels_file_;
	map<int, string> image_labels_;
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};

#endif // !GHOSTNET_H
