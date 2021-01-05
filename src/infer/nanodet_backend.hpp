
#ifndef NANODET_BACKEND_HPP
#define NANODET_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

using namespace std;
using namespace cv;

namespace TRTInfer {

	class NanoDetBackend : public Backend {
	public:
		NanoDetBackend(int max_objs = 100, CUStream stream = nullptr);

		void postProcessGPU(shared_ptr<Tensor> cls_tensor, shared_ptr<Tensor> loc_tensor, int stride, 
			Size netInputSize, float threshold, int num_classes, vector<vector<ccutil::BBox>> &bboxs, int reg_max_);


	private:
		int max_objs_;
		//vector<vector<ccutil::BBox>> outputs_;
	};
};

#endif  // NANODET_BACKEND_HPP