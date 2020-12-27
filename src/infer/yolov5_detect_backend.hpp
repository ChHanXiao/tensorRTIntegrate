
#ifndef YOLOV5_DETECT_BACKEND_HPP
#define YOLOV5_DETECT_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

using namespace std;
using namespace cv;

namespace TRTInfer {

	class YOLOv5DetectBackend : public Backend{
	public:
		YOLOv5DetectBackend(float obj_threshold, int num_classes, int max_objs, CUStream stream = nullptr);

		const vector<vector<ccutil::BBox>>& forwardGPU(
			shared_ptr<Tensor> output1, shared_ptr<Tensor> output2, shared_ptr<Tensor> output3, 
			vector<Size> imagesSize, Size inputSize);

		void decode_yolov5(
			const shared_ptr<TRTInfer::Tensor>& tensor, int stride, const vector<pair<float, float>>& anchors);
	private:

		float obj_threshold_ = 0.3;
		int num_classes_ = 80;
		int max_objs_ = 1000;
		int strides_[3] = { 8, 16, 32 };
		vector<pair<float, float >> anchor_grid_8 = { {10.000000, 13.000000}, {16.000000, 30.000000}, {33.000000, 23.000000} };
		vector<pair<float, float >> anchor_grid_16 = { {30.000000, 61.000000}, {62.000000, 45.000000}, {59.000000, 119.000000} };
		vector<pair<float, float >> anchor_grid_32 = { {116.000000, 90.000000}, {156.000000, 198.000000}, {373.000000, 326.000000} };
		vector<vector<ccutil::BBox>> outputs_;
	};
};

#endif  // YOLOV5_DETECT_BACKEND_HPP