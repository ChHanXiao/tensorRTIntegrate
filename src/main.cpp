#include <fstream>
#include <filesystem>
#include "common/cc_util.hpp"
#include "infer/trt_infer.hpp"

#include "core/face/face_engine.h"
#include "core/object/object_engine.h"
#include "core/classifier/ghostnet.h"
#include "core/object/centernet/centernet.h"
#include "core/object/nanodet/nanodet.h"
#include "core/object/yolo/yolov5.h"
#include "core/face/detect/dbface.h"
#include "core/face/detect/centerface.h"
#include "core/face/detect/retinaface.h"

using namespace std;

int TestAttribute() {
	mirror::FaceEngine face("configs/face.yaml");
	cv::Mat image = cv::imread("imgs/face2.jpg");
	std::vector<ccutil::FaceBox> faceloc;
	face.DetectFace(image, &faceloc);
	for (int i = 0; i < faceloc.size(); ++i) {
		auto& obj = faceloc[i];
		vector<cv::Point2f> keypoints;
		for (int i = 0; i < 5; i++)
		{
			keypoints.emplace_back(obj.landmark[i]);
		}
		cv::Mat img_face = image(obj.box()).clone();
		imwrite(ccutil::format("results/%d.img_face.jpg", i), img_face);
		cv::Mat face_aligned;
		face.AlignFace(image, keypoints, &face_aligned);
		vector<cv::Point2f> keypoints106;
		face.ExtractKeypoints(face_aligned, &keypoints106);
		for (const auto &point : keypoints106)
		{
			cv::circle(face_aligned, point, 1, cv::Scalar(200, 160, 75), -1, cv::LINE_8, 0);
		}
		vector<float> facefeature;
		face.ExtractFeature(face_aligned, &facefeature);

		ccutil::FaceAttribute attributes;
		face.AttrGenderAge(face_aligned, &attributes);
		string attrGenderAge = ccutil::format("%d/%d", attributes.gender, attributes.age);
		cv::putText(face_aligned, attrGenderAge, cv::Point(10, 30), 0, 0.7, cv::Scalar(0, 0, 255), 1, 16);
		imwrite(ccutil::format("results/%d.face_aligned.jpg", i), face_aligned);
		ccutil::drawbbox(image, obj);
		for (auto k : obj.landmark) {
			cv::circle(image, k, 3, cv::Scalar(0, 0, 255), -1, 16);
		}
	}
	imwrite(ccutil::format("results/face_det.jpg"), image);
	return 0;
}

int TestRecognizer() {
	mirror::FaceEngine face("configs/face.yaml");
	std::string src_path = "./imgs/recognizer/samples/";
	std::vector<cv::String> file_vec;
	cv::glob(src_path + "*.jpg", file_vec);
	face.Load();
	face.Clear();
	for (auto imgpath : file_vec) {
		cv::Mat image = cv::imread(imgpath);
		vector<float> facefeature;
		face.ExtractFeature(image, &facefeature);
		int pos = imgpath.find_last_of('\\','/');
		string nameid(imgpath.substr(pos + 1));
		face.Insert(facefeature, nameid);
	}
	face.Save();
	string imgpath = "imgs/recognizer/test2.jpg";
	cv::Mat imageid = cv::imread(imgpath);
	int pos = imgpath.find_last_of('/');
	string imgid(imgpath.substr(pos + 1));
	vector<float> facefeature;
	face.ExtractFeature(imageid, &facefeature);
	ccutil::QueryResult IDresult;
	face.QueryTop(facefeature, &IDresult);
	cout << "img:" << imgid << ",--->" << IDresult.name_<< ",sim:"<< IDresult.sim_ << endl;

	return 0;
}

int TestObject() {
#if 0
	cv::Mat image = cv::imread("imgs/obj2.jpg");	
	mirror::ObjectEngine object("configs/detect.yaml");
	std::vector<ccutil::BBox> result;
	object.DetectObject(image, &result);
	

	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/object.jpg"), image);
#else
	std::vector<cv::Mat> images{ cv::imread("imgs/zidane.jpg"), cv::imread("imgs/obj2.jpg"), cv::imread("imgs/bus.jpg") }; 
	mirror::ObjectEngine object("configs/detect.yaml");
	std::vector<std::vector<ccutil::BBox>> results;
	object.DetectObjectBatch(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.object.jpg", j), images[j]);
	}

#endif
	return 0;
}

int TestClassify() {

#if 0
 	cv::Mat image = cv::imread("imgs/train.jpg");
	GhostNet ghostnet("configs/classify/ghostnet.yaml");
	int result;
	ghostnet.EngineInference(image, &result);

	auto image_labels_ = ccutil::readImageNetLabel("./configs/classify/label.name");
 	auto result_name = image_labels_[result];
	INFO("results = %s", result_name.c_str());
	cv::putText(image, result_name, cv::Point(20, 100), 0, 2, cv::Scalar(0, 0, 255), 3, 16);
	imwrite(ccutil::format("results/prediction.jpg"), image);
#else
	std::vector<cv::Mat> images{ cv::imread("imgs/train.jpg"),cv::imread("imgs/dog.jpg") ,cv::imread("imgs/cat.jpg") ,cv::imread("imgs/eagle.jpg") };
	GhostNet ghostnet("configs/classify/ghostnet-d.yaml");
	vector<int> results;
	ghostnet.EngineInferenceOptim(images, &results);
	auto image_labels_ = ccutil::readImageNetLabel("./configs/classify/label.name");
		
	for (int j = 0; j < images.size(); ++j) {
		auto result_name = image_labels_[results[j]];
		INFO("results = %s", result_name.c_str());
		cv::putText(images[j], result_name, cv::Point(20, 100), 0, 2, cv::Scalar(0, 0, 255), 3, 16);
		imwrite(ccutil::format("results/%d.prediction.jpg", j), images[j]);
	}
#endif
	return 0;
}

int TestNanoDet() {

#if 0
	cv::Mat image = cv::imread("imgs/www.jpg");
	NanoDet object("configs/detect/nanodet.yaml");
	std::vector<ccutil::BBox> result;
	object.EngineInference(image, &result);


	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/object.jpg"), image);
#else
	std::vector<cv::Mat> images{ cv::imread("imgs/zidane.jpg") };
	NanoDet object("configs/detect/nanodet.yaml");
	std::vector<std::vector<ccutil::BBox>> results;
	object.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.object.jpg", j), images[j]);
	}

#endif


	return 0;
}

int TestCenterNet() {

#if 0
	cv::Mat image = cv::imread("imgs/obj1.jpg");
	CenterNet object("configs/detect/centernet.yaml");
	std::vector<ccutil::BBox> result;
	object.EngineInference(image, &result);


	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/object.jpg"), image);

#else
	std::vector<cv::Mat> images{ cv::imread("imgs/obj1.jpg"),cv::imread("imgs/obj2.jpg"),cv::imread("imgs/giraffe.jpg") };
	CenterNet object("configs/detect/centernet.yaml");
	std::vector<std::vector<ccutil::BBox>> results;
	object.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.object.jpg", j), images[j]);
	}

#endif
	return 0;
}

int TestYOLO() {

#if 0
	cv::Mat image = cv::imread("imgs/obj1.jpg");
	YOLOv5 object("configs/detect/yolov5s.yaml");
	std::vector<ccutil::BBox> result;
	object.EngineInference(image, &result);

	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
	}
	imwrite(ccutil::format("results/object.jpg"), image);

#else
	std::vector<cv::Mat> images{ cv::imread("imgs/obj1.jpg"),cv::imread("imgs/obj2.jpg"),cv::imread("imgs/giraffe.jpg") };
	YOLOv5 object("configs/detect/yolov5s.yaml");
	std::vector<std::vector<ccutil::BBox>> results;
	object.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
		}
		imwrite(ccutil::format("results/%d.object.jpg", j), images[j]);
	}

#endif
	return 0;
}

int TestDBFace() {
#if 1
	cv::Mat image = cv::imread("imgs/face1.jpg");
	DBFace face("configs/face/dbface.yaml");
	std::vector<ccutil::FaceBox> result;
	face.EngineInference(image, &result);


	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
		for (int k = 0; k < 5; k++)
		{
			cv::circle(image, obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
		}
	}
	imwrite(ccutil::format("results/face.jpg"), image);

#else
	std::vector<cv::Mat> images{ cv::imread("imgs/face1.jpg"),cv::imread("imgs/face2.jpg"),cv::imread("imgs/face3.jpg") };
	DBFace face("configs/face/dbface.yaml");
	std::vector<std::vector<ccutil::FaceBox>> results;
	face.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
			for (int k = 0; k < 5; k++)
			{
				cv::circle(images[j], obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
			}
		}
		imwrite(ccutil::format("results/%d.face.jpg", j), images[j]);
	}

#endif
	return 0;
}

int TestCenterFace() {
#if 1
	cv::Mat image = cv::imread("imgs/face1.jpg");
	CenterFace face("configs/face/centerface.yaml");
	std::vector<ccutil::FaceBox> result;
	face.EngineInference(image, &result);


	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
		for (int k = 0; k < 5; k++)
		{
			cv::circle(image, obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
		}
	}
	imwrite(ccutil::format("results/face.jpg"), image);

#else
	std::vector<cv::Mat> images{ cv::imread("imgs/face1.jpg"),cv::imread("imgs/face2.jpg"),cv::imread("imgs/face3.jpg") };
	CenterFace face("configs/face/centerface.yaml");
	std::vector<std::vector<ccutil::FaceBox>> results;
	face.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
			for (int k = 0; k < 5; k++)
			{
				cv::circle(images[j], obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
			}
		}
		imwrite(ccutil::format("results/%d.face.jpg", j), images[j]);
	}

#endif
	return 0;
}

int TestRetinaFace() {
#if 1
	cv::Mat image = cv::imread("imgs/face1.jpg");
	RetinaFace face("configs/face/retinaface.yaml");
	std::vector<ccutil::FaceBox> result;
	face.EngineInference(image, &result);


	for (int i = 0; i < result.size(); ++i) {
		auto& obj = result[i];
		ccutil::drawbbox(image, obj);
		for (int k = 0; k < 5; k++)
		{
			cv::circle(image, obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
		}
	}
	imwrite(ccutil::format("results/face.jpg"), image);

#else
	std::vector<cv::Mat> images{ cv::imread("imgs/face1.jpg"),cv::imread("imgs/face2.jpg"),cv::imread("imgs/face3.jpg") };
	RetinaFace face("configs/face/retinaface.yaml");
	std::vector<std::vector<ccutil::FaceBox>> results;
	face.EngineInferenceOptim(images, &results);

	for (int j = 0; j < images.size(); ++j) {
		auto& objs = results[j];
		INFO("objs.length = %d", objs.size());
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			ccutil::drawbbox(images[j], obj);
			for (int k = 0; k < 5; k++)
			{
				cv::circle(images[j], obj.landmark[k], 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
			}
		}
		imwrite(ccutil::format("results/%d.face.jpg", j), images[j]);
	}

#endif
	return 0;
}


int main() {

	TestCenterNet();


	INFO("done.");

	return 0;
}