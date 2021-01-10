#pragma once

#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <cc_util.hpp>
#include <yaml-cpp/yaml.h>

using namespace std;

namespace mirror {

	class FaceEngine {
	public:

		FaceEngine();
		FaceEngine(const string &config_file);
		~FaceEngine();

		int DetectFace(const cv::Mat& img_src, vector<ccutil::FaceBox>* faces);
		int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints);
		int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);
		int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);
		int AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes);

	private:
		class Impl;
		Impl* impl_;
	};

}




#endif // !FACE_ENGINE_H
