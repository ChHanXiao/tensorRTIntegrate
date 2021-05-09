#include "retinalp.h"

RetinaLP::RetinaLP() {};

RetinaLP::RetinaLP(const string& config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["retinalp"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	anchor_ = config["anchors"].as<vector<vector<float>>>();
	strides_ = config["strides"].as< std::vector<int>>();
	obj_threshold_ = config["obj_threshold"].as<float>();
	nms_threshold_ = config["nms_threshold"].as<float>();
	max_objs_ = config["max_objs"].as<int>();
	num_classes_ = config["num_classes"].as<int>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();

	head_out_ = { "output1", "output2", "output3 " };
	
	LoadEngine();
	GenerateAnchors();
}

RetinaLP::~RetinaLP() {};

void RetinaLP::GenerateAnchors() {
	Size netInputSize = engine_->input()->size();
	total_pix_feature_ = 0;
	for (int s = 0; s < strides_.size(); ++s) {
		int stride = strides_[s];
		int height = netInputSize.height / stride;
		int width = netInputSize.width / stride;
		total_pix_feature_ += height * width * anchor_[s].size();
	}
	anchors_matrix_ = cv::Mat(total_pix_feature_, 4, CV_32FC1);

	int line = 0;
	for (int s = 0; s < strides_.size(); ++s) {
		int stride = strides_[s];
		int height = netInputSize.height / stride;
		int width = netInputSize.width / stride;
		for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
				for (int a = 0; a < anchor_[s].size(); ++a) {
					auto *row = anchors_matrix_.ptr<float>(line);
					row[0] = (w + 0.5) * stride;
					row[1] = (h + 0.5) * stride;
					row[2] = anchor_[s][a];
					row[3] = anchor_[s][a];
					line++;
				}
			}
		}
	}
}


void RetinaLP::postProcessCPU(const shared_ptr<TRTInfer::Tensor>& conf,
	const shared_ptr<TRTInfer::Tensor>& offset, 
	const shared_ptr<TRTInfer::Tensor>& landmark,
	Mat anchors_matrix, float threshold,
	vector<ccutil::LPRBox>& bboxs) {

	int batchSize = conf->num();
	int total_pix = conf->height();
	assert(total_pix == anchors_matrix.rows);
	for (int indx = 0; indx < total_pix; ++indx) {
		float* score = conf->cpu<float>(0, 0, indx, 1);
		if (*score >= threshold) {

			auto *current_row = anchors_matrix.ptr<float>(indx);
			float cx_a = current_row[0];
			float cy_a = current_row[1];
			float w_a = current_row[2];
			float h_a = current_row[3];
			float* loc_x = offset->cpu<float>(0, 0, indx, 0);
			float* loc_y = offset->cpu<float>(0, 0, indx, 1);
			float* loc_w = offset->cpu<float>(0, 0, indx, 2);
			float* loc_h = offset->cpu<float>(0, 0, indx, 3);
			float cx_b = cx_a + *loc_x * 0.1 * w_a;
			float cy_b = cy_a + *loc_y * 0.1 * h_a;
			float w_b = w_a * expf(*loc_w * 0.2);
			float h_b = h_a * expf(*loc_h * 0.2);
			float x = cx_b - w_b * 0.5;
			float y = cy_b - h_b * 0.5;
			float r = cx_b + w_b * 0.5;
			float b = cy_b + h_b * 0.5;
			ccutil::LPRBox box(ccutil::BBox(x, y, r, b, *score, 0));
			if (box.area() > 0)
				for (int k = 0; k < 4; ++k) {
					float landmark_x = cx_a + 0.1*landmark->at<float>(0, 0, indx, k * 2) * w_a;
					float landmark_y = cy_a + 0.1*landmark->at<float>(0, 0, indx, k * 2 + 1) * h_a;
					box.landmark[k] = Point2f(landmark_x, landmark_y);
				}
			bboxs.push_back(box);
		}
	}
}

int RetinaLP::EngineInference(const Mat& image, vector<ccutil::LPRBox>* result){

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return -1;
	}
	vector<ccutil::LPRBox>& lpbboxs = *result;

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
	auto outconf = engine_->tensor("output2");
	auto outoffset = engine_->tensor("output1");
	auto outlandmark = engine_->tensor("output3");

	postProcessCPU(outconf, outoffset, outlandmark, anchors_matrix_, obj_threshold_, lpbboxs);
	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = lpbboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	PostProcess(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}

int RetinaLP::EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::LPRBox>>* result) {

	if (engine_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return -1;
	}
	vector<vector<ccutil::LPRBox>>& lpbboxs = *result;
	lpbboxs.resize(images.size());

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

	auto outconf = engine_->tensor("output2");
	auto outoffset = engine_->tensor("output1");
	auto outlandmark = engine_->tensor("output3");

	TRTInfer::RetinaLPBackend detectBackend(max_objs_, engine_->getCUStream());
	detectBackend.postProcessGPU(outconf, outoffset, outlandmark, anchors_matrix_, obj_threshold_, lpbboxs);

	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	for (int i = 0; i < lpbboxs.size(); ++i) {
		auto& objs = lpbboxs[i];
		objs = ccutil::nms(objs, nms_threshold_);
		PostProcess(objs, imagesSize[i], netInputSize);
	}
	INFO("nms time cost = %f", time_nms.end());

	return 0;

}

