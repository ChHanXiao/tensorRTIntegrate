#include "yolov5.h"

YOLOv5::YOLOv5(const string &config_file){
	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["yolov5s"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	labels_file_ = config["labels_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_minDim_ = config["input_minDim"].as<std::vector<int>>();
	input_optDim_ = config["input_optDim"].as<std::vector<int>>();
	input_maxDim_ = config["input_maxDim"].as<std::vector<int>>();
	strides_ = config["strides"].as<std::vector<int>>();
	num_anchors_ = config["num_anchors"].as<std::vector<int>>();
	anchor_grid_ = config["anchors"].as<vector<vector<vector<float>>>>();
	obj_threshold_ = config["obj_threshold"].as<float>();
	nms_threshold_ = config["nms_threshold"].as<float>();
	max_objs_ = config["max_objs"].as<int>();
	num_classes_ = config["num_classes"].as<int>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	detect_labels_ = ccutil::readCOCOLabel(labels_file_);
	assert(strides_.size() == num_anchors_.size());
	assert(num_classes_ == detect_labels_.size());

	LoadEngine();
}

YOLOv5::~YOLOv5(){}

float desigmoid(float sigmoid_value) {
	return -log(1 / sigmoid_value - 1);
}

float sigmoid(float value) {
	return 1 / (1 + exp(-value));
}

struct Anchor {
	int width[9], height[9];
};

static Rect decodeBox_yolov5(float dx, float dy, float dw, float dh, float cellx, float celly, int stride, int anchorWidth, int anchorHeight, Size netSize) {

	float cx = (dx * 2 - 0.5f + cellx) * stride;
	float cy = (dy * 2 - 0.5f + celly) * stride;
	float w = pow(dw * 2, 2) * anchorWidth;
	float h = pow(dh * 2, 2) * anchorHeight;
	float x = (cx - w * 0.5f);
	float y = (cy - h * 0.5f);
	float r = (cx + w * 0.5f);
	float b = (cy + h * 0.5f);

	return Rect(Point(x, y), Point(r + 1, b + 1));
}

void forwardCPU(const shared_ptr<TRTInfer::Tensor>& tensor, int stride, float threshold, int num_classes,
	const vector<vector<float>>& anchors, vector<ccutil::BBox> &bboxs, Size netInputSize) {
	int batch = tensor->num();
	int tensor_channel = tensor->channel();
	int tensor_width = tensor->width();
	int tensor_height = tensor->height();
	int area = tensor_width * tensor_height;
	Anchor anchor;
	for (int i = 0; i < anchors.size(); ++i) {
		anchor.width[i] = anchors[i][0];
		anchor.height[i] = anchors[i][1];
	}

	float threshold_desigmoid = desigmoid(threshold);
	for (int conf_channel = 0; conf_channel < anchors.size(); conf_channel++) {
		int conf_offset = 4 + conf_channel * (num_classes + 5);
		for (int i = 0; i < tensor_height; ++i) {
			for (int j = 0; j < tensor_width; ++j) {
				int inner_offset = i * tensor_height + j;
				float* ptr = tensor->cpu<float>(0, conf_offset) + inner_offset;
				if (*ptr < threshold_desigmoid) {
					continue;
				}
				float obj_confidence = sigmoid(*ptr);
				float* pclasses = ptr + area;
				float max_class_confidence = *pclasses;
				int max_classes = 0;
				pclasses += area;
				for (int k = 1; k < num_classes; ++k, pclasses += area) {
					if (*pclasses > max_class_confidence) {
						max_classes = k;
						max_class_confidence = *pclasses;
					}
				}
				max_class_confidence = sigmoid(max_class_confidence) * obj_confidence;
				if (max_class_confidence < threshold)
					continue;
				float* pbbox = ptr - 4 * area;
				float dx = sigmoid(*pbbox);  pbbox += area;
				float dy = sigmoid(*pbbox);  pbbox += area;
				float dw = sigmoid(*pbbox);  pbbox += area;
				float dh = sigmoid(*pbbox);  pbbox += area;

				ccutil::BBox box = decodeBox_yolov5(dx, dy, dw, dh, j, i, stride, anchor.width[conf_channel], anchor.height[conf_channel], netInputSize);
				box.label = max_classes;
				box.score = max_class_confidence;
				if (box.area() > 0)
					bboxs.push_back(box);
			}
		}
	}
}

vector<ccutil::BBox> YOLOv5::EngineInference(const Mat& image) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure, model is nullptr");
		return vector<ccutil::BBox>();
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
	vector<ccutil::BBox> bboxs;
	for (int i = 0; i < engine_->outputNum(); i++) {
		auto output = engine_->output(i);
		forwardCPU(output, strides_[i], obj_threshold_, num_classes_, anchor_grid_[i], bboxs, netInputSize);
	}
	auto& objs = bboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	outPutBox(objs, imageSize, netInputSize);
	INFO("decode time cost = %f", time_decode.end());
	return bboxs;
}

vector<vector<ccutil::BBox>> YOLOv5::EngineInferenceOptim(const vector<Mat>& images) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure call, model is nullptr");
		return vector<vector<ccutil::BBox>>();
	}

	engine_->input()->resize(images.size());
	Size netInputSize = engine_->input()->size();
	vector<Size> imagesSize;

	for (int i = 0; i < images.size(); i++) {
		preprocessImageToTensor(images[i], i, engine_->input());
		imagesSize.emplace_back(images[i].size());
	}
	engine_->forward(false);

	vector<vector<ccutil::BBox>> bboxs;
	bboxs.resize(images.size());
	TRTInfer::YOLOv5DetectBackend detectBackend(max_objs_, engine_->getCUStream());
	for (int i = 0; i < engine_->outputNum(); i++) {
		auto output = engine_->output(i);
		detectBackend.forwardGPU(output, strides_[i], obj_threshold_, num_classes_, anchor_grid_[i], bboxs, netInputSize);
	}

	for (int i = 0; i < bboxs.size(); ++i) {
		auto& objs = bboxs[i];
		objs = ccutil::nms(objs, nms_threshold_);
		outPutBox(objs, imagesSize[i], netInputSize);
	}
	return bboxs;
}

