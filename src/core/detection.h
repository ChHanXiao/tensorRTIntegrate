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

	typedef struct HeadInfo
	{
		std::string cls_layer;
		std::string dis_layer;
		int stride;
	};

	class Detection {

	public:
		Detection();
		~Detection();
		void LoadEngine();
		void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
		void outPutBox(vector<ccutil::BBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize = 15 * 15);

	public:
		string model_name_;
		string onnx_file_;
		string engine_file_;
		string labels_file_;
		shared_ptr<TRTInfer::Engine> engine_;
		vector<string> head_out_;
		int maxBatchSize_;
		vector<vector<int>> input_Dim_;

		float obj_threshold_;
		float nms_threshold_;
		int num_classes_;
		int max_objs_;
		vector<float> mean_;
		vector<float> std_;

	};
}

#endif