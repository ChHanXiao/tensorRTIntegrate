#include <fstream>
#include "common/cc_util.hpp"
#include "infer/trt_infer.hpp"
#include "common/json.hpp"

#include "core/yolov5.h"

int WriteJsonToFile(const char* filename)  
{
	Json::FastWriter writer;
	Json::Value root;
	root["name"] = "lenet";
	root["model_file"] = "./models/classifier/mnist_net.pt";
	root["onnx_file"] = "./models/classifier/mnist_net.onnx";
	root["engine_file"] = "./models/classifier/mnist_net.trt";
	root["INPUT_SIZE"].append(10);
	root["INPUT_SIZE"].append(1);
	root["INPUT_SIZE"].append(28);
	root["INPUT_SIZE"].append(28);
	std::string json_file = writer.write(root);
	std::ofstream ofs;
	ofs.open(filename);
    assert(ofs.is_open());
    ofs<<json_file;

    return 0;
}

int ReadJsonFromFile(const char* filename)  
{  
    Json::Reader reader;
    Json::Value root;
    std::ifstream is;
    is.open (filename, std::ios::binary );
    if (!reader.parse(is, root, false))
    {
		std::cout << "Error opening file\n";
        return -1;
    }
	std::string name = root["name"].asString();
	std::string model_file = root["model_file"].asString();
	std::string onnx_file = root["onnx_file"].asString();
	std::string engine_file = root["engine_file"].asString();
	int batch_size = root["INPUT_SIZE"][0].asInt();
	int input_channel = root["INPUT_SIZE"][1].asInt();
	int image_width = root["INPUT_SIZE"][2].asInt();
	int image_height = root["INPUT_SIZE"][3].asInt();
    is.close();  

    return 0;  
} 

int main() {

#if 0
	cv::Mat image = cv::imread("imgs/17790319373_bd19b24cfc_k.jpg");
	std::vector<ccutil::BBox> result;

	YOLOv5 yolov5("configs/yolov3-SPP.yaml");
	ccutil::Timer time_total;
	result = yolov5.EngineInference(image);
	INFO("total time cost = %f", time_total.end());
	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/yolov5.jpg"), image);
#else
	std::vector<cv::Mat> images{cv::imread("imgs/17790319373_bd19b24cfc_k.jpg"), cv::imread("imgs/www.jpg"),cv::imread("imgs/selfie.jpg"),cv::imread("imgs/000023.jpg") };
	std::vector<std::vector<ccutil::BBox>> results;
	YOLOv5 yolov5("configs/yolov3-SPP.yaml");
	ccutil::Timer time_total;
	results = yolov5.EngineInferenceOptim(images);
	INFO("total time cost = %f", time_total.end());
	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.yolov5.jpg", j), images[j]);
	}
#endif

//#if 1
//	cv::Mat image = cv::imread("imgs/000023.jpg");
//	std::vector<ccutil::BBox> result;
//
//	CenterNet dladcnv2("configs/yolov5.json");
//	result = dladcnv2.EngineInference(image);
//	for (int i = 0; i < result.size(); ++i) {
//		auto& obj = result[i];
//		ccutil::drawbbox(image, obj);
//	}
//	imwrite(ccutil::format("results/centernet.coco2x.dcn.jpg"), image);
//#else
//	std::vector<cv::Mat> images{cv::imread("imgs/www.jpg"), cv::imread("imgs/17790319373_bd19b24cfc_k.jpg")};
//	std::vector<std::vector<ccutil::BBox>> results;
//	CenterNet dladcnv2("configs/centernet.json");
//	ccutil::Timer time_total;
//	results = dladcnv2.EngineInferenceOptim(images);
//	INFO("total time cost = %f", time_total.end());
//	for (int j = 0; j < images.size(); ++j) {
//		auto& objs = results[j];
//		INFO("objs.length = %d", objs.size());
//		for (int i = 0; i < objs.size(); ++i) {
//			auto& obj = objs[i];
//			ccutil::drawbbox(images[j], obj);
//		}
//		imwrite(ccutil::format("results/%d.centernet.coco2x.dcn.jpg", j), images[j]);
//	}
//
//#ifdef _WIN32
//	cv::imshow("dla dcn detect 1", images[0]);
//	cv::imshow("dla dcn detect 2", images[1]);
//	cv::waitKey();
//	cv::destroyAllWindows();
//#endif
//#endif
	INFO("done.");

	return 0;
}