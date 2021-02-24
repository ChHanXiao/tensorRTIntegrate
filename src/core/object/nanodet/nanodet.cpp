#include "NanoDet.h"

NanoDet::NanoDet(const string &config_file) {

	YAML::Node root = YAML::LoadFile(config_file);
	YAML::Node config = root["nanodet"];
	onnx_file_ = config["onnx_file"].as<std::string>();
	engine_file_ = config["engine_file"].as<std::string>();
	labels_file_ = config["labels_file"].as<std::string>();
	maxBatchSize_ = config["maxBatchSize"].as<int>();
	input_Dim_ = config["input_Dim"].as<std::vector<std::vector<int>>>();
	strides_ = config["strides"].as<std::vector<int>>();
	reg_max_ = config["reg_max"].as<int>();
	obj_threshold_ = config["obj_threshold"].as<float>();
	nms_threshold_ = config["nms_threshold"].as<float>();
	max_objs_ = config["max_objs"].as<int>();
	num_classes_ = config["num_classes"].as<int>();
	mean_ = config["mean"].as<std::vector<float>>();
	std_ = config["std"].as<std::vector<float>>();
	scale_ = config["scale"].as<float>();
	detect_labels_ = ccutil::readCOCOLabel(labels_file_);

	assert(num_classes_ == detect_labels_.size());

	heads_info_ = {
		// cls_pred|dis_pred|stride
			{"output1", "output4",    8},
			{"output2", "output5",   16},
			{"output3", "output6",   32},
	};
	head_out_ = { "output1", "output2", "output3", "output4", "output5", "output6" };

	LoadEngine();
}

NanoDet::~NanoDet(){}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	const _Tp alpha = *std::max_element(src, src + length);
	_Tp denominator{ 0 };

	for (int i = 0; i < length; ++i) {
		dst[i] = exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}


static Rect decodeBox_nanodet(const float*& pdis_pred, float cellx, float celly, int stride, int reg_max) {

	float ct_x = (cellx + 0.5) * stride;
	float ct_y = (celly + 0.5) * stride;
	std::vector<float> dis_pred;
	dis_pred.resize(4);
	for (int i = 0; i < 4; i++)
	{
		float dis = 0;
		float* dis_after_sm = new float[reg_max + 1];
		activation_function_softmax(pdis_pred + i * (reg_max + 1), dis_after_sm, reg_max + 1);
		for (int j = 0; j < reg_max + 1; j++)
		{
			dis += j * dis_after_sm[j];
		}
		dis *= stride;
		dis_pred[i] = dis;
		delete[] dis_after_sm;
	}
	float x = (ct_x - dis_pred[0]);
	float y = (ct_y - dis_pred[1]);
	float r = (ct_x + dis_pred[2]);
	float b = (ct_y + dis_pred[3]);

	return Rect(Point(x, y), Point(r + 1, b + 1));
}

void NanoDet::postProcessCPU(const shared_ptr<TRTInfer::Tensor>& cls_tensor, const shared_ptr<TRTInfer::Tensor>& loc_tensor,
	int stride, Size netInputSize, float threshold, int num_classes, vector<ccutil::BBox> &bboxs, int reg_max_) {

	int batchSize = cls_tensor->num();
	int tensor_channel = cls_tensor->channel();
	int feature_area = cls_tensor->height();
	int cls_num = cls_tensor->width();
	int loc_dis = loc_tensor->width();
	int feature_h = netInputSize.height / stride;
	int feature_w = netInputSize.width / stride;
	for (int idx = 0; idx < feature_area; idx++) {
		const float* pclasses = cls_tensor->cpu<float>(0, 0, idx);
		float max_class_confidence = *pclasses;
		int max_classes = 0;
		for (int k = 0; k < num_classes; ++k, ++pclasses) {
			if (*pclasses > max_class_confidence) {
				max_classes = k;
				max_class_confidence = *pclasses;
			}
		}
		if (max_class_confidence < threshold)
			continue;

		int celly = idx / feature_w;
		int cellx = idx % feature_w;
		const float* pdis_pred = loc_tensor->cpu<float>(0, 0, idx);
		ccutil::BBox box = decodeBox_nanodet(pdis_pred, cellx, celly, stride, reg_max_);
		box.label = max_classes;
		box.score = max_class_confidence;
		if (box.area() > 0)
			bboxs.push_back(box);
	}
}

int NanoDet::EngineInference(const Mat& image, vector<ccutil::BBox>* result) {

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

	for (auto head : heads_info_) {
		auto cls_tensor = engine_->tensor(head.cls_layer);
		auto loc_tensor = engine_->tensor(head.dis_layer);
		auto stride_tmp = head.stride;
		postProcessCPU(cls_tensor, loc_tensor, stride_tmp, netInputSize, obj_threshold_, num_classes_, bboxs, reg_max_);
	}

	INFO("decode time cost = %f", time_decode.end());
	ccutil::Timer time_nms;
	auto& objs = bboxs;
	objs = ccutil::nms(objs, nms_threshold_);
	PostProcess(objs, imageSize, netInputSize);
	INFO("nms time cost = %f", time_nms.end());

	return 0;
}

// Not Support Dynamic Input 
int NanoDet::EngineInferenceOptim(const vector<Mat>& images, vector<vector<ccutil::BBox>>* result) {

	if (engine_ == nullptr) {
		INFO("EngineInference failure call, model is nullptr");
		return -1;
	}
	vector<vector<ccutil::BBox>>& bboxs = *result;
	bboxs.resize(images.size());

	ccutil::Timer time_preprocess;
	engine_->input()->resize(images.size());
	Size netInputSize = engine_->input()->size();
	vector<Size> imagesSize;
	for (int i = 0; i < images.size(); i++) {
		PrepareImage(images[i], i, engine_->input());
		imagesSize.emplace_back(images[i].size());
	}
	INFO("preprocess time cost = %f", time_preprocess.end());

	ccutil::Timer time_forward;
	engine_->forward();
	INFO("forward time cost = %f", time_forward.end());
	ccutil::Timer time_decode;

	TRTInfer::NanoDetBackend detectBackend(max_objs_, engine_->getCUStream());

	for (auto head : heads_info_) {
		auto cls_tensor = engine_->tensor(head.cls_layer);
		auto loc_tensor = engine_->tensor(head.dis_layer);
		auto stride_tmp = head.stride;
		detectBackend.postProcessGPU(cls_tensor, loc_tensor, stride_tmp, netInputSize, obj_threshold_, num_classes_, bboxs, reg_max_);
	}

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
