
#include "arcface.h"

ArcFace::ArcFace(const string &config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["arcface"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();
	//assert(num_classes_ == detect_labels_.size());

	head_out_ = { "fc1" };
	LoadEngine();
}

ArcFace::~ArcFace() {}

void ArcFace::preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

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

vector<float> ArcFace::EngineInference(const Mat &image) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return vector<float>();
	}
	ccutil::Timer time_preprocess;
	engine_->input()->resize(1);
	Size netInputSize = engine_->input()->size();
	Size imageSize = image.size();
	preprocessImageToTensor(image, 0, engine_->input());
	INFO("preprocess time cost = %f", time_preprocess.end());
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;
	vector<float> facefeature;
	auto out = engine_->tensor("fc1");
	float* out_ptr = out->cpu<float>(0, 0, 0);
	for (int i = 0; i < out->height(); ++i) {
		//cout << out_ptr[i] << endl;
		if (i % 16 == 0) {
			printf("\n");
		}
		printf("%f", out_ptr[i]);

	}
	INFO("decode time cost = %f", time_decode.end());


	return facefeature;
}

