
#include "centernet.h"

CenterNet::CenterNet() {};

CenterNet::CenterNet(const string& config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["centernet"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	strides_ = config["strides"].as<int>();
	obj_threshold_ = config["obj_threshold"].as<float>();
	nms_threshold_ = config["nms_threshold"].as<float>();
	max_objs_ = config["max_objs"].as<int>();
	num_classes_ = config["num_classes"].as<int>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();

	head_out_ = { "hm", "hm_pool", "wh", "reg"};
	LoadEngine();
}

CenterNet::~CenterNet() {};


void CenterNet::postProcessCPU(const shared_ptr<TRTInfer::Tensor>& outHM, const shared_ptr<TRTInfer::Tensor>& outHMPool,
	const shared_ptr<TRTInfer::Tensor>& outWH, const shared_ptr<TRTInfer::Tensor>& outOffset,
	int stride, float threshold, vector<ccutil::BBox>& bboxs) {

	for (int class_ = 0; class_ < outHM->channel(); ++class_) {
		for (int i = 0; i < outHM->height(); ++i) {
			float* ohmptr = outHM->cpu<float>(0, class_, i);
			float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
			for (int j = 0; j < outHM->width(); ++j) {
				if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {
					float dw = outWH->at<float>(0, 0, i, j) * stride;
					float dh = outWH->at<float>(0, 1, i, j) * stride;
					float ox = outOffset->at<float>(0, 0, i, j);
					float oy = outOffset->at<float>(0, 1, i, j);
					float x = (j + ox + 0.5) * stride - dw * 0.5;
					float y = (i + oy + 0.5) * stride - dh * 0.5;
					float r = x + dw;
					float b = y + dh;
					ccutil::BBox box(Rect(Point(x, y), Point(r + 1, b + 1)));
					box.label = class_;
					box.score = *ohmptr;
					if (box.area() > 0)
						bboxs.push_back(box);
				}
				++ohmptr;
				++ohmpoolptr;
			}
		}
	}
}

int CenterNet::EngineInference(const Mat& image, vector<ccutil::BBox>* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
	vector<ccutil::BBox>& bboxs = *result;

	ccutil::Timer time_preprocess;
	engine_->input()->resize(1);
	Size netInputSize = engine_->input()->size();
	Size imageSize = image.size();
	PrepareImage(image, 0, engine_->input());
	INFO("preprocess time cost = %f", time_preprocess.end());
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;

	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("hm_pool");
	auto outWH= engine_->tensor("wh");
	auto outOffset = engine_->tensor("reg");

	postProcessCPU(outHM, outHMPool, outWH, outOffset, strides_, obj_threshold_, bboxs);
	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = bboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	PostProcess(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}

int CenterNet::EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::BBox>>* result) {

	if (engine_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return -1;
	}
	vector<vector<ccutil::BBox>>& bboxs = *result;
	bboxs.resize(images.size());

	engine_->input()->resize(images.size());
	Size netInputSize = engine_->input()->size();
	vector<Size> imagesSize;
	for (int i = 0; i < images.size(); ++i) {
		PrepareImage(images[i], i, engine_->input());
		imagesSize.emplace_back(images[i].size());
	}
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("engine forward cost = %f", time_forward.end());
	ccutil::Timer time_decode;

	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("hm_pool");
	auto outWH = engine_->tensor("wh");
	auto outOffset = engine_->tensor("reg");
	TRTInfer::CenterNetBackend detectBackend(max_objs_, engine_->getCUStream());
	detectBackend.postProcessGPU(outHM, outHMPool, outWH, outOffset, strides_, obj_threshold_, bboxs);

	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	for (int i = 0; i < bboxs.size(); ++i) {
		auto& objs = bboxs[i];
		objs = ccutil::nms(objs, nms_threshold_);
		PostProcess(objs, imagesSize[i], netInputSize);
	}
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}
