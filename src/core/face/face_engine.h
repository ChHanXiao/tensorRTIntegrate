#pragma once

#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <cc_util.hpp>
#include <yaml-cpp/yaml.h>


namespace mirror {

	class FaceEngine {

	public:
		FaceEngine();
		FaceEngine(const std::string &config_file);
		~FaceEngine();

		int DetectFace(const cv::Mat& img_src, std::vector<ccutil::FaceBox>* faces);
		int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints);
		int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned);
		int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feature);
		int AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes);

		int Insert(const std::vector<float>& feat, const std::string& name);
		int Delete(const std::string& name);
		int64_t QueryTop(const std::vector<float>& feat, ccutil::QueryResult *query_result = nullptr);
		int Save();
		int Load();
		int Clear();

	private:
		class Impl;
		Impl* impl_;
	};

}


#endif // !FACE_ENGINE_H
