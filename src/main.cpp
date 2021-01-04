#include <fstream>
#include "common/cc_util.hpp"
#include "infer/trt_infer.hpp"
#include "common/json.hpp"

#include "core/yolov5.h"
#include "core/nanodet.h"

int main() {

#if 0
	cv::Mat image = cv::imread("imgs/17790319373_bd19b24cfc_k.jpg");
	std::vector<ccutil::BBox> result;

	NanoDet nanodet("configs/nanodet.yaml");
	ccutil::Timer time_total;
	int i = 0;
	while (i < 1000){
		result = nanodet.EngineInference(image);
		i++;
	}

	INFO("total time cost = %f", time_total.end()/1000);
	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/nanodet.jpg"), image);
#else
	//std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
	std::vector<cv::Mat> images{ cv::imread("imgs/17790319373_bd19b24cfc_k.jpg") };
	std::vector<std::vector<ccutil::BBox>> results;
	NanoDet nanodet("configs/nanodet.yaml");
	ccutil::Timer time_total;

	int i = 0;
	while (i < 1000) {
		results = nanodet.EngineInferenceOptim(images);
		i++;
	}

	INFO("total time cost = %f", time_total.end()/1000);
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

	//#ifdef _WIN32
	//	cv::imshow("detect 1", images[0]);
	//	cv::imshow("detect 2", images[1]);
	//	cv::waitKey();
	//	cv::destroyAllWindows();
	//#endif

	INFO("done.");

	return 0;
}