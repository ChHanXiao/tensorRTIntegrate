#include "object_engine.h"
#include "nanodet/nanodet.h"
#include "yolo/yolov5.h"

namespace mirror {

	class ObjectEngine::Impl {

	public:
		Impl(const std::string &config_file) {
			INFO("ObjectEngine Init Start!");
			YAML::Node root = YAML::LoadFile(config_file);
			YAML::Node config = root["detect"];
			std::string nanodet_cfg = config["nanodet"].as<std::string>();
			std::string yolo_cfg = config["yolo"].as<std::string>();
			//nanodet_detecter_ = new NanoDet(nanodet_cfg);
			yolo_detecter_ = new YOLOv5(yolo_cfg);
			initialized_ = true;
			INFO("ObjectEngine Init End!");
		}

		~Impl() {}

		inline int DetectObject(const cv::Mat& img_src, std::vector<ccutil::BBox>* objects) {
			return yolo_detecter_->EngineInference(img_src, objects);
		}
		inline int DetectObjectBatch(const std::vector<Mat>& img_src, std::vector<std::vector<ccutil::BBox>>* objects) {
			return yolo_detecter_->EngineInferenceOptim(img_src, objects);
		}

	private:
		bool initialized_ = false;
		NanoDet* nanodet_detecter_;
		YOLOv5* yolo_detecter_;

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