
#include "detection.h"

namespace ObjectDetection {

	Detection::Detection() {}
	Detection::~Detection() {}

	void Detection::PrepareImage(const Mat& image, int numIndex, const shared_ptr<TRTInfer::Tensor>& tensor) {

		int outH = tensor->height();
		int outW = tensor->width();
		float sw = outW / (float)image.cols;
		float sh = outH / (float)image.rows;
		float scale_size = std::min(sw, sh);
		cv::Mat flt_img = cv::Mat::zeros(cv::Size(outW, outH), CV_8UC3);
		cv::Mat outimage;
		cv::resize(image, outimage, cv::Size(), scale_size, scale_size);
		outimage.copyTo(flt_img(cv::Rect(0, 0, outimage.cols, outimage.rows)));
		float mean[3], std[3];
		for (int i = 0; i < 3; i++)
		{
			mean[i] = mean_[i];
			std[i] = std_[i];
		}

		tensor->setNormMatGPU(numIndex, flt_img, mean, std, scale_);
	}

	void Detection::PostProcess(vector<ccutil::BBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize) {
		float sw = netInputSize.width / (float)imageSize.width;
		float sh = netInputSize.height / (float)imageSize.height;
		float scale_size = std::min(sw, sh);

		vector<ccutil::BBox> keep;
		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			obj.x = std::max(0.0f, std::min(obj.x / scale_size, imageSize.width - 1.0f));
			obj.y = std::max(0.0f, std::min(obj.y / scale_size, imageSize.height - 1.0f));
			obj.r = std::max(0.0f, std::min(obj.r / scale_size, imageSize.width - 1.0f));
			obj.b = std::max(0.0f, std::min(obj.b / scale_size, imageSize.height - 1.0f));

			if (obj.area() > minsize)
				keep.emplace_back(obj.x, obj.y, obj.r, obj.b, obj.score, obj.label);
		}
		objs = keep;
	}
	void Detection::PostProcess(vector<ccutil::FaceBox>& objs, const Size& imageSize, const Size& netInputSize, float minsize) {
		float sw = netInputSize.width / (float)imageSize.width;
		float sh = netInputSize.height / (float)imageSize.height;
		float scale_size = std::min(sw, sh);

		vector<ccutil::FaceBox> keep;
		
 		for (int i = 0; i < objs.size(); ++i) {
			auto& obj = objs[i];
			obj.x = std::max(0.0f, std::min(obj.x / scale_size, imageSize.width - 1.0f));
			obj.y = std::max(0.0f, std::min(obj.y / scale_size, imageSize.height - 1.0f));
			obj.r = std::max(0.0f, std::min(obj.r / scale_size, imageSize.width - 1.0f));
			obj.b = std::max(0.0f, std::min(obj.b / scale_size, imageSize.height - 1.0f));

			for (auto &k : obj.landmark) {
				k.x /= scale_size;
				k.y /= scale_size;
			}

			if (obj.area() > minsize)
				keep.emplace_back(obj);
				
		}
		objs = keep;
	}

}