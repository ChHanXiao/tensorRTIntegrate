
#include "ghostnet.h"

GhostNet::GhostNet() {}

GhostNet::GhostNet(const string &config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["ghostnet"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	labels_file_ = config["labels_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();
	image_labels_ = ccutil::readImageNetLabel(labels_file_);
	//assert(num_classes_ == detect_labels_.size());
	head_out_ = { "output" };

	LoadEngine();

}

GhostNet::~GhostNet() {}

void GhostNet::PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	int outH = tensor->height();
	int outW = tensor->width();
	float sw = outW / (float)image.cols;
	float sh = outH / (float)image.rows;
	float scale_size = std::min(sw, sh);
	cv::Mat flt_img = cv::Mat::zeros(cv::Size(outW, outH), CV_8UC3);
	cv::Mat outimage;
	cv::resize(image, outimage, cv::Size(), scale_size, scale_size);
	outimage.copyTo(flt_img(cv::Rect(0, 0, outimage.cols, outimage.rows)));
	float mean[3], std[3];
	for (int i = 0; i < 3; i++)
	{
		mean[i] = mean_[i];
		std[i] = std_[i];
	}

	tensor->setNormMatGPU(numIndex, flt_img, mean, std, scale_);
}

int GhostNet::EngineInference(const Mat &image, int* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
	int& cls = *result;

	ccutil::Timer time_preprocess;
	engine_->input()->resize(1);
	PrepareImage(image, 0, engine_->input());
	INFO("preprocess time cost = %f", time_preprocess.end());
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;
	auto out = engine_->tensor("output");
	auto outSize = out->channel();
	float* out_ptr = out->cpu<float>(0, 0);
	cls = max_element(out_ptr, out_ptr + outSize) - out_ptr;
	auto label_name = image_labels_[cls];
	INFO("result_name = %s", label_name.c_str());

	INFO("decode time cost = %f", time_decode.end());

	return 0;
}

int GhostNet::EngineInferenceOptim(const vector<Mat>& images, vector<int>* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
	vector<int>& cls = *result;
	cls.resize(images.size());

	ccutil::Timer time_preprocess;
	engine_->input()->resize(images.size());
	for (int i = 0; i < images.size(); i++) {
		PrepareImage(images[i], i, engine_->input());
	}
	INFO("preprocess time cost = %f", time_preprocess.end());
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;

	auto out = engine_->tensor("output");
	auto outSize = out->channel();
	for (int i = 0; i < images.size(); ++i) {
		float* out_ptr = out->cpu<float>(i, 0);
		auto result = max_element(out_ptr, out_ptr + outSize);
		cls[i] = result - out_ptr;
		auto label_name = image_labels_[cls[i]];
		INFO("result_name = %s", label_name.c_str());
	}
	INFO("decode time cost = %f", time_decode.end());

	return 0;
}
