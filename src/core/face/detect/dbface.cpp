
#include "dbface.h"

DBFace::DBFace() {};

DBFace::DBFace(const string& config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["dbface"];
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

	head_out_ = { "hm", "pool_hm", "tlrb", "landmark"};
	LoadEngine();
}

DBFace::~DBFace() {};

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

void DBFace::postProcessCPU(const shared_ptr<TRTInfer::Tensor>& outHM, const shared_ptr<TRTInfer::Tensor>& outHMPool,
	const shared_ptr<TRTInfer::Tensor>& outTLRB, const shared_ptr<TRTInfer::Tensor>& outLandmark,
	int stride, float threshold, vector<ccutil::FaceBox>& bboxs) {

	for (int class_ = 0; class_ < outHM->channel(); ++class_) {
		for (int i = 0; i < outHM->height(); ++i) {
			float* ohmptr = outHM->cpu<float>(0, class_, i);
			float* ohmpoolptr = outHMPool->cpu<float>(0, class_, i);
			for (int j = 0; j < outHM->width(); ++j) {
				if (*ohmptr == *ohmpoolptr && *ohmpoolptr > threshold) {
					float dx = outTLRB->at<float>(0, 0, i, j) ;
					float dy = outTLRB->at<float>(0, 1, i, j) ;
					float dr = outTLRB->at<float>(0, 2, i, j) ;
					float db = outTLRB->at<float>(0, 3, i, j) ;
					float cx = j;
					float cy = i;
					float x = (cx - dx) * stride;
					float y = (cy - dy) * stride;
					float r = (cx + dr) * stride;
					float b = (cy + db) * stride;
					ccutil::FaceBox box(ccutil::BBox(x, y, r, b, *ohmptr, class_));
					if (box.area() > 0)
						for (int k = 0; k < 5; ++k) {
							float landmark_x = outLandmark->at<float>(0, k, i, j) * stride;
							float landmark_y = outLandmark->at<float>(0, k + 5, i, j) * stride;
							landmark_x = (commonExp(landmark_x) + cx) * stride;
							landmark_y = (commonExp(landmark_y) + cy) * stride;
							box.landmark[k] = Point2f(landmark_x, landmark_y);
						}
						bboxs.push_back(box);
				}
				++ohmptr;
				++ohmpoolptr;
			}
		}
	}
}

int DBFace::EngineInference(const Mat& image, vector<ccutil::FaceBox>* result) {

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
	engine_->forward(false);
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;
	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("pool_hm");
	auto outTLRB = engine_->tensor("tlrb");
	auto outLandmark = engine_->tensor("landmark");
	postProcessCPU(outHM, outHMPool, outTLRB, outLandmark, strides_, obj_threshold_, facebboxs);
	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = facebboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	PostProcess(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}

int DBFace::EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::FaceBox>>* result) {

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

	auto outHM = engine_->tensor("hm");
	auto outHMPool = engine_->tensor("pool_hm");
	auto outTLRB = engine_->tensor("tlrb");
	auto outLandmark = engine_->tensor("landmark");
	TRTInfer::DBFaceBackend detectBackend(max_objs_, engine_->getCUStream());
	detectBackend.postProcessGPU(outHM, outHMPool, outTLRB, outLandmark, strides_, obj_threshold_, facebboxs);

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
