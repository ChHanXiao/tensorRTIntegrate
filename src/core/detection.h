#pragma once

#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <cc_util.hpp>
#include "builder/trt_builder.hpp"
#include "infer/trt_infer.hpp"
#include "common/json.hpp"

using namespace std;
using namespace cv;

namespace ObjectDetection {
	class Detection {

	public:
		Detection(const string &config_file);
		~Detection();
		void LoadEngine();
		void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
		void outPutBox(vector<ccutil::BBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize = 15 * 15);

	public:
		string model_name_;
		string onnx_file_;
		string engine_file_;
		shared_ptr<TRTInfer::Engine> engine_;
		int maxBatchSize_ = 1;
		vector<int> input_minDim = { 1, 3, 640, 640 };
		vector<int> input_optDim = { 1, 3, 640, 640 };
		vector<int> input_maxDim = { 1, 3, 640, 640 };

		float obj_threshold_ = 0.3;
		float nms_threshold_ = 0.6;
		int num_classes_ = 80;
		int max_objs_ = 1000;
		float mean_[3] = { 0., 0., 0. };
		float std_[3] = { 1., 1., 1. };

	};
}

#endif