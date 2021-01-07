#include "retinaface.h"

RetinaFace::RetinaFace(const string &config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["retinaface"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	labels_file_ = config["labels_file"].as<std::string>();
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
	detect_labels_ = ccutil::readCOCOLabel(labels_file_);
	//assert(num_classes_ == detect_labels_.size());

	head_out_ = { "output1", "output2", "output3 " };
	
	LoadEngine();
	GenerateAnchors();
}

RetinaFace::~RetinaFace() {};

void RetinaFace::GenerateAnchors() {
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

void postProcessCPU(const shared_ptr<TRTInfer::Tensor>& conf,
	const shared_ptr<TRTInfer::Tensor>& offset, 
	const shared_ptr<TRTInfer::Tensor>& landmark,
	Mat anchors_matrix, float threshold,
	vector<ccutil::FaceBox> &bboxs) {

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
			ccutil::BBox box(Rect(Point(x, y), Point(r + 1, b + 1)));
			box.label = 0;
			box.score = *score;
			ccutil::FaceBox facebox(box);
			if (box.area() > 0)
				for (int k = 0; k < 5; ++k) {
					float landmark_x = cx_a + 0.1*landmark->at<float>(0, 0, indx, k * 2) * w_a;
					float landmark_y = cy_a + 0.1*landmark->at<float>(0, 0, indx, k * 2 + 1) * h_a;
					facebox.landmark[k] = Point2f(landmark_x, landmark_y);
				}
			bboxs.push_back(facebox);
		}
	}
}

vector<ccutil::FaceBox> RetinaFace::EngineInference(const Mat& image){

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return vector<ccutil::FaceBox>();
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
	vector<ccutil::FaceBox> facebboxs;
	auto outconf = engine_->tensor("output2");
	auto outoffset = engine_->tensor("output1");
	auto outlandmark = engine_->tensor("output3");
	postProcessCPU(outconf, outoffset, outlandmark, anchors_matrix_, obj_threshold_, facebboxs);
	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = facebboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	outPutBox(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return facebboxs;
}

vector<vector<ccutil::FaceBox>> RetinaFace::EngineInferenceOptim(const vector<Mat>& images) {

	if (engine_ == nullptr) {
		INFO("detectBoundingbox failure call, model is nullptr");
		return vector<vector<ccutil::FaceBox>>();
	}
	engine_->input()->resize(images.size());
	Size netInputSize = engine_->input()->size();
	vector<Size> imagesSize;
	for (int i = 0; i < images.size(); ++i) {
		preprocessImageToTensor(images[i], i, engine_->input());
		imagesSize.emplace_back(images[i].size());
	}
	ccutil::Timer time_forward;
	engine_->forward();
	INFO("engine forward cost = %f", time_forward.end());
	ccutil::Timer time_decode;
	vector<vector<ccutil::FaceBox>> facebboxs;
	facebboxs.resize(images.size());
	auto outconf = engine_->tensor("output2");
	auto outoffset = engine_->tensor("output1");
	auto outlandmark = engine_->tensor("output3");

	TRTInfer::RetinaFaceBackend detectBackend(max_objs_, engine_->getCUStream());
	detectBackend.postProcessGPU(outconf, outoffset, outlandmark, anchors_matrix_, obj_threshold_, facebboxs);

	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	for (int i = 0; i < facebboxs.size(); ++i) {
		auto& objs = facebboxs[i];
		objs = ccutil::nms(objs, nms_threshold_);
		outPutBox(objs, imagesSize[i], netInputSize);
	}
	INFO("nms time cost = %f", time_nms.end());

	return facebboxs;

}

