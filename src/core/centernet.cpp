
#include "centernet.h"

CenterNet::CenterNet(const std::string &config_file) {
	Json::Reader reader;
	Json::Value root;
	std::ifstream is_file;
	is_file.open(config_file, std::ios::binary);
	if (!reader.parse(is_file, root, false))
	{
		std::cout << "Error opening file\n";
	}
	model_name_ = root["name"].asString();
	onnx_file_ = root["onnx_file"].asString();
	engine_file_ = root["engine_file"].asString();
	batch_size_ = root["INPUT_SIZE"][0].asInt();
	input_channel_ = root["INPUT_SIZE"][1].asInt();
	image_width_ = root["INPUT_SIZE"][2].asInt();
	image_height_ = root["INPUT_SIZE"][3].asInt();
	max_objs_ = root["max_objs"].asInt();
	obj_threshold_ = root["obj_threshold"].asFloat();
	nms_threshold_ = root["nms_threshold"].asFloat();
	is_file.close();

	LoadEngine();
}

CenterNet::~CenterNet() {
};


static Rect restoreCenterNetBox(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, Size netSize, Size imageSize) {

	float scale = 0;
	if (imageSize.width >= imageSize.height)
		scale = netSize.width / (float)imageSize.width;
	else
		scale = netSize.height / (float)imageSize.height;

	float x = ((cellx + dx - dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float y = ((celly + dy - dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	float r = ((cellx + dx + dw * 0.5) * stride - netSize.width * 0.5) / scale + imageSize.width * 0.5;
	float b = ((celly + dy + dh * 0.5) * stride - netSize.height * 0.5) / scale + imageSize.height * 0.5;
	return Rect(Point(x, y), Point(r + 1, b + 1));
}

static void preprocessCenterNetImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

	int outH = tensor->height();
	int outW = tensor->width();
	float sw = outW / (float)image.cols;
	float sh = outH / (float)image.rows;
	float scale = std::min(sw, sh);

	Mat matrix = getRotationMatrix2D(Point2f(image.cols*0.5, image.rows*0.5), 0, scale);
	matrix.at<double>(0, 2) -= image.cols*0.5 - outW * 0.5;
	matrix.at<double>(1, 2) -= image.rows*0.5 - outH * 0.5;

	float mean[3] = { 0.40789654, 0.44719302, 0.47026115 };
	float std[3] = { 0.28863828, 0.27408164, 0.27809835 };

	Mat outimage;
	cv::warpAffine(image, outimage, matrix, Size(outW, outH));
	tensor->setNormMatGPU(numIndex, outimage, mean, std);
}

void CenterNet::LoadEngine() {
	
	INFO("LoadEngine...");
	if (!ccutil::exists(engine_file_)) {
		INFO("onnx to trtmodel...");
		if (!ccutil::exists(onnx_file_)) {
			INFOW("onnx file:%s not found !", onnx_file_.c_str());
			return;
		}
		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, {}, batch_size_,
			TRTBuilder::ModelSource(onnx_file_),
			engine_file_, { TRTBuilder::InputDims(3, 512, 512) }
		);
	}
	INFO("load model: %s", engine_file_.c_str());
	engine_ = TRTInfer::loadEngine(engine_file_);
}

vector<ccutil::BBox> CenterNet::EngineInference(const Mat& image) {
	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return vector<ccutil::BBox>();
	}
	preprocessCenterNetImageToTensor(image, 0, engine_->input());
	engine_->forward();

	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("hm_pool");
	auto outWH = engine_->tensor("wh");
	auto outXY = engine_->tensor("reg");
	const int stride = 4;

	vector<ccutil::BBox> bboxs;
	Size inputSize = engine_->input()->size();

	for (int class_ = 0; class_ < outHM->channel(); ++class_) {
		for (int i = 0; i < outHM->height(); ++i) {
			float* ohmptr = outHM->cpu<float>(0, class_, i);
			float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
			for (int j = 0; j < outHM->width(); ++j) {
				if (*ohmptr == *ohmpoolptr && *ohmpoolptr > obj_threshold_) {

					float dx = outXY->at<float>(0, 0, i, j);
					float dy = outXY->at<float>(0, 1, i, j);
					float dw = outWH->at<float>(0, 0, i, j);
					float dh = outWH->at<float>(0, 1, i, j);
					ccutil::BBox box = restoreCenterNetBox(dx, dy, dw, dh, j, i, stride, inputSize, image.size());
					box = box.box() & Rect(0, 0, image.cols, image.rows);
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
	return bboxs;

}

vector<vector<ccutil::BBox>> CenterNet::EngineInferenceOptim(const vector<Mat>& images) {
	if (engine_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return vector<vector<ccutil::BBox>>();
	}
	
	engine_->input()->resize(images.size());
	vector<Size> imsize;
	for (int i = 0; i < images.size(); ++i) {
		preprocessCenterNetImageToTensor(images[i], i, engine_->input());
		imsize.emplace_back(images[i].size());
	}
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("engine forward cost = %f", time_forward.end());
	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("hm_pool");
	auto outWH = engine_->tensor("wh");
	auto outXY = engine_->tensor("reg");
	TRTInfer::CTDetectBackend detectBackend(engine_->getCUStream());

	return detectBackend.forwardGPU(outHM, outHMPool, outWH, outXY, imsize, obj_threshold_, max_objs_);
}
