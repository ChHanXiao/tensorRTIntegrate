#pragma once

#ifndef GHOSTNET_H
#define GHOSTNET_H

#include "core/base/trtmodel.h"

class GhostNet : public TrtModel{

public:

	GhostNet();
	GhostNet(const string &config_file);
	~GhostNet();
	
	void PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat &image, int* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<int>* result);
public:
	string labels_file_;
	map<int, string> image_labels_;
	vector<float> mean_;
	vector<float> std_;
	float scale_;
};


#endif // !GHOSTNET_H

