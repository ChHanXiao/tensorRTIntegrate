#pragma once

#ifndef LPRNET_H
#define LPRNET_H

#include "core/base/trtmodel.h"

class LPRNet : public TrtModel {

public:

	LPRNet();
	LPRNet(const string &config_file);
	~LPRNet();
	void PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
	int EngineInference(const Mat& image, wstring* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<wstring>* result);
	wstring GreedyDecode(vector<int> preb_label, int outSize, int outNum);

public:
	vector<float> mean_;
	vector<float> std_;
	float scale_;
	const wstring CHARS_ = L"¾©»¦½òÓå¼½½úÃÉÁÉ¼ªºÚËÕÕãÍîÃö¸ÓÂ³Ô¥¶õÏæÔÁ¹ğÇí´¨¹óÔÆ²ØÉÂ¸ÊÇàÄşĞÂ0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-";
};

#endif // !LPRNET_H