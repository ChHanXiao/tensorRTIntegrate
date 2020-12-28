

#ifndef TRT_BUILDER_HPP
#define TRT_BUILDER_HPP

#include <string>
#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>

namespace TRTBuilder {

	typedef std::function<void(int current, int count, cv::Mat& inputOutput)> Int8Process;

	void setDevice(int device_id);

	enum ModelSourceType {
		ModelSourceType_FromCaffe,
		ModelSourceType_FromONNX
	};

	class ModelSource {
	public:
		ModelSource(const std::string& prototxt, const std::string& caffemodel);
		ModelSource(const std::string& onnxmodel);
		ModelSourceType type() const;
		std::string prototxt() const;
		std::string caffemodel() const;
		std::string onnxmodel() const;

	private:
		std::string prototxt_, caffemodel_;
		std::string onnxmodel_;
		ModelSourceType type_;
	};

	class InputDims {
	public:
		InputDims(int batchsize, int channels, int height, int width);

		int batchsize() const;
		int channels() const;
		int height() const;
		int width() const;

	private:
		int batchsize_, channels_, height_, width_;
	};

	enum TRTMode {
		TRTMode_FP32,
		TRTMode_FP16
	};

	const char* modeString(TRTMode type);

	bool compileTRT(
		TRTMode mode,
		const std::vector<std::string>& outputs,
		int maxBatchSize,
		const ModelSource& source,
		const std::string& savepath,
		const std::vector<std::vector<int>> inputsDimsSetup = {});
};

#endif //TRT_BUILDER_HPP