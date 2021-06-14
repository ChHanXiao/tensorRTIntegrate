#include "lpr_engine.h"
#include "detect\retinalp.h"
#include "recognize\lprnet.h"
#include "recognize\crop_lp.h"

namespace mirror {
	class LPREngine::Impl {

	public:
		Impl(const std::string &config_file) {
			INFO("LPREngine Init Start!");
			YAML::Node root = YAML::LoadFile(config_file);
			YAML::Node config = root["lpr"];
			std::string detect_cfg = config["detect"].as<std::string>();
			std::string recognize_cfg = config["recognize"].as<std::string>();

			detecter_ = new RetinaLP(detect_cfg);
			recognizer_ = new LPRNet(recognize_cfg);
			initialized_ = true;
			INFO("LPREngine Init End!");

		}
		~Impl() {}

		inline int DetectLP(const cv::Mat& img_src, std::vector<ccutil::LPRBox>* lpboxs) {
			return detecter_->EngineInference(img_src, lpboxs);
		}
		inline int CropLP(const cv::Mat& img_src,const ccutil::LPRBox lpbox, cv::Mat* img_crop) {
			return crop_lp(img_src, lpbox, img_crop);
		}
		inline int RecognizeLP(const cv::Mat& image, std::wstring* lpresult) {
			return recognizer_->EngineInference(image, lpresult);
		}
	private:
		bool initialized_ = false;
		RetinaLP* detecter_;
		LPRNet* recognizer_;
	};

	LPREngine::LPREngine() {}

	LPREngine::LPREngine(const std::string &config_file) {
		impl_ = new LPREngine::Impl(config_file);
	}
	LPREngine::~LPREngine() {
		if (impl_) {
			delete impl_;
			impl_ = nullptr;
		}
	}
	int LPREngine::DetectLP(const cv::Mat& img_src, std::vector<ccutil::LPRBox>* lpboxs) {
		return impl_->DetectLP(img_src, lpboxs);
	}

	int LPREngine::CropLP(const cv::Mat& img_src, const ccutil::LPRBox lpbox, cv::Mat* img_crop) {
		return impl_->CropLP(img_src, lpbox, img_crop);
	}

	int LPREngine::RecognizeLP(const cv::Mat& image,  std::wstring* lpresult) {
		return impl_->RecognizeLP(image, lpresult);
	}


}