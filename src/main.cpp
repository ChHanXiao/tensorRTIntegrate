#include <fstream>
#include "common/cc_util.hpp"
#include "infer/trt_infer.hpp"
#include "common/json.hpp"

#include "core/yolov5.h"
#include "core/nanodet.h"
#include "core/centerface.h"

int main() {


#if 1
	cv::Mat image = cv::imread("imgs/dog.jpg");
	std::vector<ccutil::BBox> result;

	NanoDet nanodet("configs/nanodet.yaml");
	ccutil::Timer time_total;
	result = nanodet.EngineInference(image);

	INFO("total time cost = %f", time_total.end());
	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/nanodet.jpg"), image);
#else
	//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
	std::vector<cv::Mat> images{ cv::imread("imgs/dog.jpg")};
	std::vector<std::vector<ccutil::BBox>> results;
	NanoDet nanodet("configs/nanodet.yaml");
	ccutil::Timer time_total;
	results = nanodet.EngineInferenceOptim(images);

	INFO("total time cost = %f", time_total.end());
	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.nanodet.jpg", j), images[j]);
	}
#endif


//#if 0
//	cv::Mat image = cv::imread("imgs/selfie.jpg");
//	std::vector<ccutil::FaceBox> result;
//
//	CenterFace nanodet("configs/centerface.yaml");
//	ccutil::Timer time_total;
//	result = nanodet.EngineInference(image);
//
//	INFO("total time cost = %f", time_total.end()/1000);
//	for (int i = 0; i < result.size(); ++i) {
//		auto& obj = result[i];
//		ccutil::drawbbox(image, obj);
//		for (auto k : obj.landmark) {
//			cv::circle(image, k, 3, Scalar(0, 0, 255), -1, 16);
//		}
//	}
//	imwrite(ccutil::format("results/nanodet.jpg"), image);
//#else
//	//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
//	std::vector<cv::Mat> images{ cv::imread("imgs/test1.jpg"), cv::imread("imgs/selfie.jpg"), cv::imread("imgs/00001.jpg") }; 
//	//std::vector<std::vector<ccutil::BBox>> results;
//	std::vector<std::vector<ccutil::FaceBox>> results;
//	CenterFace nanodet("configs/centerface.yaml");
//	ccutil::Timer time_total;
//
//	results = nanodet.EngineInferenceOptim(images);
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
//		imwrite(ccutil::format("results/%d.nanodet.jpg", j), images[j]);
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