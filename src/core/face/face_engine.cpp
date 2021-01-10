#include "face_engine.h"
#include "detect/retinaface.h"
#include "detect/centerface.h"
#include "landmark/face_alignment.h"
#include "attribute/gender_age.h"
#include "recognize/aligner.h"
#include "recognize/arcface.h"

namespace mirror {

	class FaceEngine::Impl {
	public:
		Impl(const string &config_file) {

			INFO("FaceEngine Init Start!");
			YAML::Node root = YAML::LoadFile(config_file);
			YAML::Node config = root["face"];
			string detect_cfg = config["detect"].as<std::string>();
			string face_alignment_cfg = config["face_alignment"].as<std::string>();
			string gender_age_cfg = config["gender_age"].as<std::string>();
			string recognizer_cfg = config["recognizer"].as<std::string>();

			detecter_ = RetinaFace(detect_cfg);
			landmarker_ = FaceAlignment(face_alignment_cfg);
			genderAge_ = GenderAge(gender_age_cfg);
			aligner_ = Aligner();
			recognizer_ = ArcFace(recognizer_cfg);
			initialized_ = true;
			INFO("FaceEngine Init End!");
		}

		~Impl(){}

		inline int DetectFace(const cv::Mat& img_src, vector<ccutil::FaceBox>* faces) {
			return detecter_.EngineInference(img_src, faces);
		}
		inline int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints) {
			return landmarker_.EngineInference(img_face, keypoints);
		}
		inline int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
			return aligner_.AlignFace(img_src, keypoints, face_aligned);
		}
		inline int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
			return recognizer_.EngineInference(img_face, feat);
		}
		inline int AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes) {
			return genderAge_.EngineInference(image, attributes);
		}	

	private:
		bool initialized_ = false;
		RetinaFace detecter_;
		FaceAlignment landmarker_;
		GenderAge genderAge_;
		Aligner aligner_;
		ArcFace recognizer_;
	};
	FaceEngine::FaceEngine() {}

	FaceEngine::FaceEngine(const string &config_file) {
		impl_ = new FaceEngine::Impl(config_file);
	}

	FaceEngine::~FaceEngine() {
		if (impl_) {
			delete impl_;
			impl_ = nullptr;
		}
	}

	int FaceEngine::DetectFace(const cv::Mat& img_src, vector<ccutil::FaceBox>* faces) {
		return impl_->DetectFace(img_src, faces);
	}

	int FaceEngine::ExtractKeypoints(const cv::Mat& img_face, vector<cv::Point2f>* keypoints) {
		return impl_->ExtractKeypoints(img_face, keypoints);
	}

	int FaceEngine::AlignFace(const cv::Mat& img_src, const vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
		return impl_->AlignFace(img_src, keypoints, face_aligned);
	}

	int FaceEngine::ExtractFeature(const cv::Mat& img_face, vector<float>* feat) {
		return impl_->ExtractFeature(img_face, feat);
	}

	int FaceEngine::AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes) {
		return impl_->AttrGenderAge(image, attributes);
	}



}
