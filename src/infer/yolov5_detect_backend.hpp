
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
		YOLOv5DetectBackend(vector<vector<vector<float>>> anchor_grid, vector<int> strides, float obj_threshold, int num_classes, int max_objs, CUStream stream = nullptr);

		const vector<vector<ccutil::BBox>>& forwardGPU(
			shared_ptr<Tensor> output1, shared_ptr<Tensor> output2, shared_ptr<Tensor> output3, 
			vector<Size> imagesSize, Size inputSize);

		void decode_yolov5(
			const shared_ptr<TRTInfer::Tensor>& tensor, int stride, const vector<vector<float>>& anchors);
	private:

		float obj_threshold_;
		int num_classes_;
		int max_objs_;
		vector<int> strides_;
		vector<vector<vector<float>>> anchor_grid_;

		//vector<pair<float, float >> anchor_grid_8 = { {13.0, 17.0}, {31.0, 25.0}, {24.0, 51.0}, {61.0, 45.0} };
		//vector<pair<float, float >> anchor_grid_16 = { {48.0, 102.0}, {119.0, 96.0}, {97.0, 189.0}, {217.0, 184.0} };
		//vector<pair<float, float >> anchor_grid_32 = { {171.0, 384.0}, {324.0, 451.0}, {616.0, 618.0}, {800.0, 800.0} };

		vector<vector<ccutil::BBox>> outputs_;
	};
};

#endif  // YOLOV5_DETECT_BACKEND_HPP