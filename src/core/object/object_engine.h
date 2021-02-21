#pragma once

#ifndef OBJECTER_H
#define OBJECTER_H

#include <cc_util.hpp>
#include <yaml-cpp/yaml.h>

namespace mirror {

	class ObjectEngine {

	public:
		ObjectEngine();
		ObjectEngine(const std::string &config_file);
		~ObjectEngine();

		int DetectObject(const cv::Mat& img_src, std::vector<ccutil::BBox>* objects = nullptr);
		int DetectObjectBatch(const std::vector<cv::Mat>& img_src, std::vector<std::vector<ccutil::BBox>>* objects = nullptr);
		
	private:
		class Impl;
		Impl* impl_;

	};

}

#endif // !OBJECTER_H
