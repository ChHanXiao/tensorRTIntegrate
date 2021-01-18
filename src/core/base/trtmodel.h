#pragma once

#ifndef TRTMODEL_H
#define TRTMODEL_H

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <cc_util.hpp>
//#include <register_factory.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"

using namespace std;
using namespace cv;

class TrtModel{

public:
	TrtModel();
	~TrtModel();
	void LoadEngine();
public:
	string model_name_;
	string onnx_file_;
	string engine_file_;
	shared_ptr<TRTInfer::Engine> engine_;
	vector<string> head_out_;
	int maxBatchSize_;
	vector<vector<int>> input_Dim_;

};


#endif // !TRTMODEL_H
