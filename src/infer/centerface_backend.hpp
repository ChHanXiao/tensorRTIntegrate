
#ifndef CENTERFACE_BACKEND_HPP
#define CENTERFACE_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

using namespace std;
using namespace cv;

namespace TRTInfer {

	class CenterFaceBackend : public Backend{
	public:
		CenterFaceBackend(int max_objs = 100, CUStream stream = nullptr);

		void postProcessGPU(shared_ptr<Tensor> heatmap, shared_ptr<Tensor> scalewh, 
			shared_ptr<Tensor> offset, shared_ptr<Tensor> landmark,
			int stride, float threshold, vector<vector<ccutil::FaceBox>> &bboxs);

	private:
		int max_objs_;
	};
};

#endif  // CENTERFACE_BACKEND_HPP