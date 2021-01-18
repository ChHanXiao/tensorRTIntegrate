#pragma once

#ifndef YOLOV5_H
#define YOLOV5_H

#include "infer/yolov5_detect_backend.hpp"
#include "core/base/detection.h"

using namespace ObjectDetection;

class YOLOv5 : public Detection {
public:
	YOLOv5(const string &config_file);
	~YOLOv5();
	void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& tensor, int stride, float threshold, int num_classes,
		const vector<vector<float>>& anchors, vector<ccutil::BBox> &bboxs);
	int EngineInference(const Mat& image, vector<ccutil::BBox>* result);
	int EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::BBox>>* result);

private:
	vector<int> strides_;
	vector<int> num_anchors_;
	map<int, string> detect_labels_;
	vector<vector<vector<float>>> anchor_grid_ ;
};


#endif // !YOLOV5_H