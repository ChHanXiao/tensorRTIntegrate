
#include "trtmodel.h"

TrtModel::TrtModel() {}
TrtModel::~TrtModel() {}

void TrtModel::LoadEngine() {

	INFO("LoadEngine...");
	if (!ccutil::exists(engine_file_)) {
		INFO("onnx to trtmodel...");
		if (!ccutil::exists(onnx_file_)) {
			INFOW("onnx file:%s not found !", onnx_file_.c_str());
			return;
		}
		TRTBuilder::compileTRT(
			TRTBuilder::TRTMode_FP32, head_out_, maxBatchSize_,
			TRTBuilder::ModelSource(onnx_file_), engine_file_,
			input_Dim_
		);
	}
	INFO("load model: %s", engine_file_.c_str());
	engine_ = TRTInfer::loadEngine(engine_file_);
}
