
#include "centerface.h"

CenterFace::CenterFace() {};

CenterFace::CenterFace(const string& config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["centerface"];
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

	head_out_ = { "537", "538", "539", "540"};
	LoadEngine();
}

CenterFace::~CenterFace() {};

static float commonExp(float value) {

	float gate = 1;
	float base = exp(gate);
	if (fabs(value) < gate)
		return value * base;

	if (value > 0) {
		return exp(value);
	}
	else {
		return -exp(-value);
	}
}

void CenterFace::postProcessCPU(const shared_ptr<TRTInfer::Tensor>& heatmap, const shared_ptr<TRTInfer::Tensor>& wh,
	const shared_ptr<TRTInfer::Tensor>& offset, const shared_ptr<TRTInfer::Tensor>& landmark,
	int stride, float threshold, vector<ccutil::FaceBox>& bboxs) {

	for (int class_ = 0; class_ < heatmap->channel(); ++class_) {
		for (int i = 0; i < heatmap->height(); ++i) {
			float* ohmptr = heatmap->cpu<float>(0, class_, i);
			for (int j = 0; j < heatmap->width(); ++j) {
				if (*ohmptr > threshold) {
					float dh = commonExp(wh->at<float>(0, 0, i, j)) * stride;
					float dw = commonExp(wh->at<float>(0, 1, i, j)) * stride;
					float oy = offset->at<float>(0, 0, i, j);
					float ox = offset->at<float>(0, 1, i, j);

					float x = (j + ox + 0.5) * stride - dw * 0.5;
					float y = (i + oy + 0.5) * stride - dh * 0.5;
					float r = x + dw;
					float b = y + dh;
					ccutil::FaceBox box(ccutil::BBox(x, y, r, b, *ohmptr, class_));
					if (box.area() > 0)
						for (int k = 0; k < 5; ++k) {
							float landmark_x = box.x + landmark->at<float>(0, k * 2 + 1, i, j) * box.width();
							float landmark_y = box.y + landmark->at<float>(0, k * 2, i, j) * box.height();
							box.landmark[k] = Point2f(landmark_x, landmark_y);
						}
						bboxs.push_back(box);
				}
				++ohmptr;
			}
		}
	}
}

int CenterFace::EngineInference(const Mat& image, vector<ccutil::FaceBox>* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
	vector<ccutil::FaceBox>& facebboxs = *result;

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

	auto outhm = engine_->tensor("537");
	auto outwh = engine_->tensor("538");
	auto outoffset = engine_->tensor("539");
	auto outlandmark = engine_->tensor("540");
	postProcessCPU(outhm, outwh, outoffset, outlandmark, strides_, obj_threshold_, facebboxs);
	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = facebboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	PostProcess(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}

int CenterFace::EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::FaceBox>>* result) {

	if (engine_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return -1;
	}
	vector<vector<ccutil::FaceBox>>& facebboxs = *result;
	facebboxs.resize(images.size());

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

	auto outhm = engine_->tensor("537");
	auto outwh = engine_->tensor("538");
	auto outoffset = engine_->tensor("539");
	auto outlandmark = engine_->tensor("540");
	TRTInfer::CenterFaceBackend detectBackend(max_objs_, engine_->getCUStream());
	detectBackend.postProcessGPU(outhm, outwh, outoffset, outlandmark, strides_, obj_threshold_, facebboxs);

	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	for (int i = 0; i < facebboxs.size(); ++i) {
		auto& objs = facebboxs[i];
		objs = ccutil::nms(objs, nms_threshold_);
		PostProcess(objs, imagesSize[i], netInputSize);
	}
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}
