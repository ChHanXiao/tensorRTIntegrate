#pragma once

#ifndef RETINALP_BACKEND_HPP
#define RETINALP_BACKEND_HPP

#include <cc_util.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include "trt_backend.hpp"

using namespace std;
using namespace cv;

namespace TRTInfer {

	class RetinaLPBackend : public Backend {
	public:
		RetinaLPBackend(int max_objs = 100, CUStream stream = nullptr);

		void postProcessGPU(shared_ptr<Tensor> conf, shared_ptr<Tensor> offset,
			shared_ptr<Tensor> landmark, Mat anchors_matrix_,
			float threshold, vector<vector<ccutil::LPRBox>> &bboxs);

	private:
		int max_objs_;
	};
};

#endif  // RETINALP_BACKEND_HPP