
#include "detection.h"

namespace ObjectDetection {
	Detection::Detection(const string &config_file)
	{
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
		maxBatchSize_ = root["maxBatchSize"].asInt();

		input_minDim[0] = root["input_minDim"][0].asInt();
		input_minDim[1] = root["input_minDim"][1].asInt();
		input_minDim[2] = root["input_minDim"][2].asInt();
		input_minDim[3] = root["input_minDim"][3].asInt();

		input_optDim[0] = root["input_optDim"][0].asInt();
		input_optDim[1] = root["input_optDim"][1].asInt();
		input_optDim[2] = root["input_optDim"][2].asInt();
		input_optDim[3] = root["input_optDim"][3].asInt();

		input_maxDim[0] = root["input_maxDim"][0].asInt();
		input_maxDim[1] = root["input_maxDim"][1].asInt();
		input_maxDim[2] = root["input_maxDim"][2].asInt();
		input_maxDim[3] = root["input_maxDim"][3].asInt();

		max_objs_ = root["max_objs"].asInt();
		obj_threshold_ = root["obj_threshold"].asFloat();
		nms_threshold_ = root["nms_threshold"].asFloat();

		num_classes_ = root["num_classes"].asInt();
		mean_[0] = root["mean"][0].asFloat();
		mean_[1] = root["mean"][1].asFloat();
		mean_[2] = root["mean"][2].asFloat();

		std_[0] = root["std"][0].asFloat();
		std_[1] = root["std"][1].asFloat();
		std_[2] = root["std"][2].asFloat();

		is_file.close();
		LoadEngine();
	}

	Detection::~Detection(){}

	void Detection::LoadEngine() {

		INFO("LoadEngine...");
		if (!ccutil::exists(engine_file_)) {
			INFO("onnx to trtmodel...");
			if (!ccutil::exists(onnx_file_)) {
				INFOW("onnx file:%s not found !", onnx_file_.c_str());
				return;
			}
			TRTBuilder::compileTRT(
				TRTBuilder::TRTMode_FP16, {}, maxBatchSize_,
				TRTBuilder::ModelSource(onnx_file_), engine_file_,
				{ input_minDim, input_optDim, input_maxDim }
			);
		}
		INFO("load model: %s", engine_file_.c_str());
		engine_ = TRTInfer::loadEngine(engine_file_);
	}

	void Detection::preprocessImageToTensor(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

		int outH = tensor->height();
		int outW = tensor->width();
		float sw = outW / (float)image.cols;
		float sh = outH / (float)image.rows;
		float scale = std::min(sw, sh);

		cv::Mat flt_img = cv::Mat::zeros(cv::Size(outW, outH), CV_8UC3);
		cv::Mat outimage;
		cv::resize(image, outimage, cv::Size(), scale, scale);
		outimage.copyTo(flt_img(cv::Rect(0, 0, outimage.cols, outimage.rows)));

		tensor->setNormMatGPU(numIndex, flt_img, mean_, std_);
	}

	void Detection::outPutBox(vector<ccutil::BBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize) {
		float sw = netInputSize.width / (float)imageSize.width;
		float sh = netInputSize.height / (float)imageSize.height;
		float scale = std::min(sw, sh);

		vector<ccutil::BBox> keep;
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			obj.x = std::max(0.0f, std::min(obj.x / scale, imageSize.width - 1.0f));
			obj.y = std::max(0.0f, std::min(obj.y / scale, imageSize.height - 1.0f));
			obj.r = std::max(0.0f, std::min(obj.r / scale, imageSize.width - 1.0f));
			obj.b = std::max(0.0f, std::min(obj.b / scale, imageSize.height - 1.0f));

			if (obj.area() > minsize)
				keep.emplace_back(obj.x, obj.y, obj.r, obj.b, obj.score, obj.label);
		}
		objs = keep;
	}

}