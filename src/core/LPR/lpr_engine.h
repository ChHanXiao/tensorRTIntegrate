#pragma once

#ifndef LPR_ENGINE_H
#define LPR_ENGINE_H

#include <cc_util.hpp>
#include <yaml-cpp/yaml.h>

namespace mirror {
	class LPREngine {

	public:
		LPREngine();
		LPREngine(const std::string &config_file);
		~LPREngine();
		int DetectLP(const cv::Mat& img_src, std::vector<ccutil::LPRBox>* lpboxs);
		int CropLP(const cv::Mat& img_src, const ccutil::LPRBox lpbox, cv::Mat* img_crop);
		int RecognizeLP(const cv::Mat& image, std::wstring* lpresult);
	
	private:
		class Impl;
		Impl* impl_;
	};

}

#endif // !LPR_ENGINE_H