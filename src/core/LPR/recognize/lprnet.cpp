
#include "lprnet.h"

LPRNet::LPRNet() {}

LPRNet::LPRNet(const string &config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["lprnet"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();

	head_out_ = { "output" };
	LoadEngine();
}

LPRNet::~LPRNet() {}

void LPRNet::PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	int outH = tensor->height();
	int outW = tensor->width();

	cv::Mat outimage;
	cv::resize(image, outimage, cv::Size(outW, outH));
	float mean[3], std[3];
	for (int i = 0; i < 3; i++)
	{
		mean[i] = mean_[i];
		std[i] = std_[i];
	}

	tensor->setNormMatGPU(numIndex, outimage, mean, std, scale_);
}

wstring LPRNet::GreedyDecode(vector<int> preb_label, int outNum, int outSize) {

	vector<int> no_repeat_blank_label;
	int pre_c = preb_label[0];
	int last = outSize - 1;
	if (pre_c != last)
	{
		no_repeat_blank_label.push_back(pre_c);
	}
	int c = 0;
	for (int i = 0; i < outNum; i++)
	{
		c = preb_label[i];
		if ((pre_c == c) || (c == last))
		{
			if (c == last)
			{
				pre_c = c;
			}
			continue;
		}
		no_repeat_blank_label.push_back(c);
		pre_c = c;
	}
	int len_s = no_repeat_blank_label.size();
	wstring results;
	for (int i = 0; i < len_s; i++)
	{
		results.push_back(CHARS_[no_repeat_blank_label[i]]);
	}
	return results;
}

int LPRNet::EngineInference(const Mat &image, wstring* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
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
	auto out = engine_->tensor("output");
	auto outNum = out->channel();
	auto outSize = out->height();

	vector<int> preb_label;
	preb_label.resize(outNum);

	float* out_ptr_tmp = out->cpu<float>(0, 0);

	for (int i = 0; i < outNum; i++) {
		float* out_ptr = out->cpu<float>(0, i);
		preb_label[i] = max_element(out_ptr, out_ptr + outSize) - out_ptr;
	}
	*result = GreedyDecode(preb_label, outNum, outSize);

	INFO("decode time cost = %f", time_decode.end());

	return 0;
}

int LPRNet::EngineInferenceOptim(const vector<Mat>& images, vector<wstring>* result) {

	if (engine_ == nullptr) {
		INFO("EngineInferenceOptim failure, model is nullptr");
		return -1;
	}
	vector<wstring>& lprresults = *result;
	ccutil::Timer time_preprocess;
	lprresults.resize(images.size());
	engine_->input()->resize(images.size());
	vector<Size> imagesSize;
	for (int i = 0; i < images.size(); ++i) {
		PrepareImage(images[i], i, engine_->input());
		imagesSize.emplace_back(images[i].size());
	}
	INFO("preprocess time cost = %f", time_preprocess.end());

	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());

	ccutil::Timer time_decode;
	auto out = engine_->tensor("output");
	for (int idx = 0; idx < images.size(); ++idx) {
		auto outNum = out->channel();
		auto outSize = out->height();
		vector<int> preb_label;
		preb_label.resize(outNum);

		for (int i = 0; i < outNum; i++) {
			float* out_ptr = out->cpu<float>(idx, i);
			preb_label[i] = max_element(out_ptr, out_ptr + outSize) - out_ptr;
		}
		lprresults[idx] = GreedyDecode(preb_label, outNum, outSize);
	}
 	INFO("decode time cost = %f", time_decode.end());

	return 0;
}
