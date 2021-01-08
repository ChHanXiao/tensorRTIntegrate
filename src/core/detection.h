#pragma once

#ifndef DETECTION_H
#define DETECTION_H

#include "trtmodel.h"

namespace ObjectDetection {

	class Detection : public TrtModel {

	public:
		Detection();
		~Detection();
		void preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor);
		void outPutBox(vector<ccutil::BBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize = 15 * 15);
		void outPutBox(vector<ccutil::FaceBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize = 1);

	public:
		string labels_file_;

		float obj_threshold_;
		float nms_threshold_;
		int num_classes_;
		int max_objs_;
		vector<float> mean_;
		vector<float> std_;
		float scale_;
	};
}

#endif // !DETECTION_H