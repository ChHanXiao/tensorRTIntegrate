#include "face_engine.h"
#include "detect/retinaface.h"
#include "detect/centerface.h"
#include "landmark/face_alignment.h"
#include "attribute/gender_age.h"
#include "recognize/aligner.h"
#include "recognize/arcface.h"
#include "database/face_database.h"

namespace mirror {

	class FaceEngine::Impl {
	public:
		Impl(const std::string &config_file) {

			INFO("FaceEngine Init Start!");
			YAML::Node root = YAML::LoadFile(config_file);
			YAML::Node config = root["face"];
			std::string detect_cfg = config["detect"].as<std::string>();
			std::string face_alignment_cfg = config["face_alignment"].as<std::string>();
			std::string gender_age_cfg = config["gender_age"].as<std::string>();
			std::string recognize_cfg = config["recognize"].as<std::string>();
			db_name_ = config["db_path"].as<std::string>();
			detecter_ = new RetinaFace(detect_cfg);
			landmarker_ = new FaceAlignment(face_alignment_cfg);
			genderAge_ = new GenderAge(gender_age_cfg);
			aligner_ = new Aligner();
			recognizer_ = new ArcFace(recognize_cfg);
			database_ = new FaceDatabase();
			initialized_ = true;
			INFO("FaceEngine Init End!");
		}

		~Impl(){}

		inline int DetectFace(const cv::Mat& img_src, std::vector<ccutil::FaceBox>* faces) {
			return detecter_->EngineInference(img_src, faces);
		}
		inline int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints) {
			return landmarker_->EngineInference(img_face, keypoints);
		}
		inline int AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
			return aligner_->AlignFace(img_src, keypoints, face_aligned);
		}
		inline int ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
			return recognizer_->EngineInference(img_face, feat);
		}
		inline int AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes) {
			return genderAge_->EngineInference(image, attributes);
		}	
		inline int Insert(const std::vector<float>& feat, const std::string& name) {
			return database_->Insert(feat, name);
		}
		inline int Delete(const std::string& name) {
			return database_->Delete(name);
		}
		inline int64_t QueryTop(const std::vector<float>& feat, ccutil::QueryResult *query_result = nullptr) {
			return database_->QueryTop(feat, query_result);
		}
		inline int Save() {
			return  database_->Save(db_name_.c_str());
		}
		inline int Load() {
			return database_->Load(db_name_.c_str());
		}
		inline bool Clear() {
			return database_->Clear();
		}
	private:
		bool initialized_ = false;
		std::string db_name_;
		RetinaFace* detecter_;
		FaceAlignment* landmarker_;
		GenderAge* genderAge_;
		Aligner* aligner_;
		ArcFace* recognizer_;
		FaceDatabase* database_;
	};
	FaceEngine::FaceEngine() {}

	FaceEngine::FaceEngine(const std::string &config_file) {
		impl_ = new FaceEngine::Impl(config_file);
	}

	FaceEngine::~FaceEngine() {
		if (impl_) {
			delete impl_;
			impl_ = nullptr;
		}
	}

	int FaceEngine::DetectFace(const cv::Mat& img_src, std::vector<ccutil::FaceBox>* faces) {
		return impl_->DetectFace(img_src, faces);
	}

	int FaceEngine::ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints) {
		return impl_->ExtractKeypoints(img_face, keypoints);
	}

	int FaceEngine::AlignFace(const cv::Mat& img_src, const std::vector<cv::Point2f>& keypoints, cv::Mat* face_aligned) {
		return impl_->AlignFace(img_src, keypoints, face_aligned);
	}

	int FaceEngine::ExtractFeature(const cv::Mat& img_face, std::vector<float>* feat) {
		return impl_->ExtractFeature(img_face, feat);
	}

	int FaceEngine::AttrGenderAge(const cv::Mat &image, ccutil::FaceAttribute* attributes) {
		return impl_->AttrGenderAge(image, attributes);
	}

	int FaceEngine::Insert(const std::vector<float>& feat, const std::string& name) {
		return impl_->Insert(feat, name);
	}

	int FaceEngine::Delete(const std::string& name) {
		return impl_->Delete(name);
	}

	int64_t FaceEngine::QueryTop(const std::vector<float>& feat, ccutil::QueryResult* query_result) {
		return impl_->QueryTop(feat, query_result);
	}

	int FaceEngine::Save() {
		return impl_->Save();
	}

	int FaceEngine::Load() {
		return impl_->Load();
	}
	
	bool FaceEngine::Clear() {
		return impl_->Clear();
	}

}
