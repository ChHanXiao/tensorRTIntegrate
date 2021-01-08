#include <fstream>
#include "common/cc_util.hpp"
#include "infer/trt_infer.hpp"
#include "common/json.hpp"

#include "core/yolov5.h"
#include "core/nanodet.h"
#include "core/centerface.h"
#include "core/retinaface.h"
#include "core/arcface.h"
#include "core/aligner.h"
#include "core/ghostnet.h"

int main() {

#if 0
	cv::Mat image = cv::imread("imgs/train.jpg");

	GhostNet ghostnet("configs/ghostnet.yaml");
	ccutil::Timer time_total;
	auto result = ghostnet.EngineInference(image);
	auto image_labels_ = ccutil::readImageNetLabel("./configs/label.txt");
	auto result_name = image_labels_[result];
	INFO("results = %s", result_name.c_str());
	cv::putText(image, result_name, cv::Point(20, 100), 0, 2, cv::Scalar(0,0,255), 3, 16);
	imwrite(ccutil::format("results/prediction.jpg"), image);

#else
		//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
	std::vector<cv::Mat> images{ cv::imread("imgs/train.jpg"),cv::imread("imgs/dog.jpg") ,cv::imread("imgs/cat.jpg") };
	GhostNet ghostnet("configs/ghostnet.yaml");
	ccutil::Timer time_total;
	auto results = ghostnet.EngineInferenceOptim(images);
	auto image_labels_ = ccutil::readImageNetLabel("./configs/label.txt");
	
	INFO("total time cost = %f", time_total.end());
	for (int j = 0; j < images.size(); ++j) {
		auto result_name = image_labels_[results[j]];
		INFO("results = %s", result_name.c_str());
		cv::putText(images[j], result_name, cv::Point(20, 100), 0, 2, cv::Scalar(0, 0, 255), 3, 16);
		imwrite(ccutil::format("results/%d.prediction.jpg", j), images[j]);
	}


#endif

	//cv::Mat image = cv::imread("imgs/00001.jpg");
	//std::vector<ccutil::FaceBox> Faceloc;
	//RetinaFace retinaface("configs/retinaface.yaml");
	//ArcFace arcface("configs/arcface.yaml");
	//Aligner aligner;
	//Faceloc = retinaface.EngineInference(image);
	//for (int i = 0; i < Faceloc.size(); ++i) {
	//	auto& obj = Faceloc[i];
	//	vector<cv::Point2f> keypoints;
	//	for (int i = 0; i < 5; i++)
	//	{
	//		keypoints.emplace_back(obj.landmark[i]);
	//	}
	//	cv::Mat img_face = image(obj.box()).clone();
	//	imwrite(ccutil::format("results/%d.img_face.jpg", i), img_face);
	//	cv::Mat face_aligned;
	//	aligner.AlignFace(image, keypoints, &face_aligned);
	//	imwrite(ccutil::format("results/%d.face_aligned.jpg", i), face_aligned);
	//	auto Facefeature = arcface.EngineInference(face_aligned);
	//}



//#if 1
//	cv::Mat image = cv::imread("imgs/00001.jpg");
//	std::vector<ccutil::FaceBox> result;
//
//	RetinaFace retinaface("configs/retinaface.yaml");
//	ccutil::Timer time_total;
//	result = retinaface.EngineInference(image);
//
//	INFO("total time cost = %f", time_total.end());
//	for (int i = 0; i < result.size(); ++i) {
//		auto& obj = result[i];
//		ccutil::drawbbox(image, obj);
//		for (auto k : obj.landmark) {
//			cv::circle(image, k, 3, Scalar(0, 0, 255), -1, 16);
//		}
//	}
//	imwrite(ccutil::format("results/prediction.jpg"), image);
//
//#else
//	//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
//	std::vector<cv::Mat> images{ cv::imread("imgs/test1.jpg"), cv::imread("imgs/selfie.jpg"), cv::imread("imgs/00001.jpg") }; 
//	//std::vector<std::vector<ccutil::BBox>> results;
//	std::vector<std::vector<ccutil::FaceBox>> results;
//	RetinaFace retinaface("configs/retinaface.yaml");
//	ccutil::Timer time_total;
//
//	results = retinaface.EngineInferenceOptim(images);
//
//	INFO("total time cost = %f", time_total.end());
//	for (int j = 0; j < images.size(); ++j) {
//		auto& objs = results[j];
//		INFO("objs.length = %d", objs.size());
//		for (int i = 0; i < objs.size(); ++i) {
//			auto& obj = objs[i];
//			ccutil::drawbbox(images[j], obj);
//			for (auto k : obj.landmark) {
//				cv::circle(images[j], k, 3, Scalar(0, 0, 255), -1, 16);
//			}
//		}
//		imwrite(ccutil::format("results/%d.prediction.jpg", j), images[j]);
//	}
//#endif


//#if 1
//	cv::Mat image = cv::imread("imgs/giraffe.jpg");
//	std::vector<ccutil::BBox> result;
//
//	NanoDet nanodet("configs/nanodet.yaml");
//	ccutil::Timer time_total;
//	result = nanodet.EngineInference(image);
//
//	INFO("total time cost = %f", time_total.end());
//	for (int i = 0; i < result.size(); ++i) {
//		auto& obj = result[i];
//		ccutil::drawbbox(image, obj);
//	}
//	imwrite(ccutil::format("results/prediction.jpg"), image);
//#else
//	//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
//	std::vector<cv::Mat> images{ cv::imread("imgs/www.jpg")};
//	std::vector<std::vector<ccutil::BBox>> results;
//	NanoDet nanodet("configs/nanodet.yaml");
//	ccutil::Timer time_total;
//	results = nanodet.EngineInferenceOptim(images);
//
//	INFO("total time cost = %f", time_total.end());
//	for (int j = 0; j < images.size(); ++j) {
//		auto& objs = results[j];
//		INFO("objs.length = %d", objs.size());
//		for (int i = 0; i < objs.size(); ++i) {
//			auto& obj = objs[i];
//			ccutil::drawbbox(images[j], obj);
//		}
//		imwrite(ccutil::format("results/%d.prediction.jpg", j), images[j]);
//	}
//#endif


	//#ifdef _WIN32
	//	cv::imshow("detect 1", images[0]);
	//	cv::imshow("detect 2", images[1]);
	//	cv::waitKey();
	//	cv::destroyAllWindows();
	//#endif

	INFO("done.");

	return 0;
}