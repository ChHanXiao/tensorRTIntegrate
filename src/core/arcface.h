#pragma once

#ifndef ARCFACE_H
#define ARCFACE_H

#include "detection.h"
#include "yaml-cpp/yaml.h"

using namespace ObjectDetection;

class ArcFace : public Detection {
public:
	ArcFace(const string &config_file);
	~ArcFace();

	vector<float> EngineInference(const Mat &image);
private:

};

#endif