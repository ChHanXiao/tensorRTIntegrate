
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
		YOLOv5DetectBackend(int max_objs, CUStream stream = nullptr);

		void forwardGPU(shared_ptr<Tensor> output, int stride, float threshold, int num_classes,
			const vector<vector<float>>& anchors, vector<vector<ccutil::BBox>> &bboxs, Size netInputSize);

	private:
		int max_objs_;
		//vector<vector<ccutil::BBox>> outputs_;
	};
};

#endif  // YOLOV5_DETECT_BACKEND_HPP