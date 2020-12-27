#pragma once

#ifndef CENTERNET_H
#define CENTERNET_H

#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include "infer/ct_detect_backend.hpp"
#include "common/json.hpp"

using namespace std;
using namespace cv;


class CenterNet {
public:
	CenterNet(const std::string &config_file);
	~CenterNet();
	void LoadEngine();
	vector<ccutil::BBox> EngineInference(const Mat& image);
	vector<vector<ccutil::BBox>> EngineInferenceOptim(const vector<Mat>& images);
private:
	string model_name_;
	string onnx_file_;
	string engine_file_;
	shared_ptr<TRTInfer::Engine> engine_;
	int batch_size_ = 1;
	int input_channel_ = 3;
	int image_width_ = 512;
	int image_height_ = 512;
	int max_objs_ = 100;
	float obj_threshold_ = 0.3;
	float nms_threshold_ = 0.6;
};


#endif