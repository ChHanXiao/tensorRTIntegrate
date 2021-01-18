#include "object_engine.h"
#include "nanodet/nanodet.h"
#include "yolo/yolov5.h"

namespace mirror {

	class ObjectEngine::Impl {

	public:
		Impl(const std::string &config_file) {
			YAML::Node root = YAML::LoadFile(config_file);
			YAML::Node config = root["detect"];
			std::string detecter_type = config["type"].as<std::string>();
			std::string detecter_cfg = config["config"].as<std::string>();

			detecter_ = new YOLOv5(detecter_cfg);
			initialized_ = true;

		}

		~Impl() {
			delete detecter_;
			detecter_ = nullptr;
		}

		inline int DetectObject(const cv::Mat& img_src, std::vector<ccutil::BBox>* objects) {
			return detecter_->EngineInference(img_src, objects);
		}
		inline int DetectObjectBatch(const std::vector<Mat>& img_src, std::vector<std::vector<ccutil::BBox>>* objects) {
			return detecter_->EngineInferenceOptim(img_src, objects);
		}

	private:
		bool initialized_ = false;
		YOLOv5* detecter_;

	};

	ObjectEngine::ObjectEngine() {}

	ObjectEngine::ObjectEngine(const std::string &config_file) {
		impl_ = new ObjectEngine::Impl(config_file);
	}

	ObjectEngine::~ObjectEngine() {
		if (impl_) {
			delete impl_;
			impl_ = nullptr;
		}
	}

	int ObjectEngine::DetectObject(const cv::Mat& img_src, std::vector<ccutil::BBox>* objects) {
		return impl_->DetectObject(img_src, objects);
	}

	int ObjectEngine::DetectObjectBatch(const std::vector<cv::Mat>& img_src, std::vector<std::vector<ccutil::BBox>>* objects) {
		return impl_->DetectObjectBatch(img_src, objects);
	}


}