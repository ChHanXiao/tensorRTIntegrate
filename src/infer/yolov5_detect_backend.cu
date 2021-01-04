

#include "yolov5_detect_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	YOLOv5DetectBackend::YOLOv5DetectBackend(int max_objs, CUStream stream) :Backend(stream) {

		this->max_objs_ = max_objs;
	}

	static __device__ float sigmoid(float value) {
		return 1 / (1 + exp(-value));
	}

	static float desigmoid(float val) {
		return -log(1 / val - 1);
	}

	struct Anchor {
		int width[9], height[9];
	};

	static __global__ void decode_native_impl(float* data,
		int width, int height, int stride, float threshold, float threshold_desigmoid, int num_classes,
		Anchor anchor, ccutil::BBox* output, int* counter, int area, int maxobjs, int edge) {

		KERNEL_POSITION;

		int inner_offset = position % area;
		int a = position / area;
		float* ptr = data + (a * (num_classes + 5) + 4) * area + inner_offset;

		if (*ptr < threshold_desigmoid)
			return;

		float obj_confidence = sigmoid(*ptr);
		float* pclasses = ptr + area;
		float max_class_confidence = *pclasses;
		int max_classes = 0;
		pclasses += area;

		for (int j = 1; j < num_classes; ++j, pclasses += area) {
			if (*pclasses > max_class_confidence) {
				max_classes = j;
				max_class_confidence = *pclasses;
			}
		}

		max_class_confidence = sigmoid(max_class_confidence) * obj_confidence;
		if (max_class_confidence < threshold)
			return;

		int index = atomicAdd(counter, 1);
		if (index >= maxobjs)
			return;

		float* pbbox = ptr - 4 * area;
		float dx = sigmoid(*pbbox);  pbbox += area;
		float dy = sigmoid(*pbbox);  pbbox += area;
		float dw = sigmoid(*pbbox);  pbbox += area;
		float dh = sigmoid(*pbbox);  pbbox += area;

		int cell_x = position % width;
		int cell_y = (position / width) % height;
		float cx = (dx * 2 - 0.5f + cell_x) * stride;
		float cy = (dy * 2 - 0.5f + cell_y) * stride;
		float w = pow(dw * 2, 2) * anchor.width[a];
		float h = pow(dh * 2, 2) * anchor.height[a];
		float x = cx - w * 0.5f;
		float y = cy - h * 0.5f;
		float r = cx + w * 0.5f;
		float b = cy + h * 0.5f;
		ccutil::BBox& box = output[index];
		box.x = x;
		box.y = y;
		box.r = r;
		box.b = b;
		box.label = max_classes;
		box.score = max_class_confidence;
	}


	void YOLOv5DetectBackend::postProcessGPU(shared_ptr<Tensor> tensor, int stride, float threshold, int num_classes,
		const vector<vector<float>>& anchors, vector<vector<ccutil::BBox>> &bboxs) {

		float threshold_desigmoid = desigmoid(threshold);
		int tensor_width = tensor->width();
		int tensor_height = tensor->height();
		int batchSize = tensor->num();
		size_t objsStoreSize = max_objs_ * sizeof(ccutil::BBox) + sizeof(int);
		int area = tensor_width * tensor_height;
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();
		int job_count = area * anchors.size();
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);
		Anchor anchor;

		for (int i = 0; i < anchors.size(); ++i) {
			anchor.width[i] = anchors[i][0];
			anchor.height[i] = anchors[i][1];
		}

		for (int n = 0; n < batchSize; ++n) {

			int* counter = (int*)gpuPtrInput;
			ccutil::BBox* bboxptr = (ccutil::BBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(counter, 0, sizeof(int), stream);
			decode_native_impl << < grid, block, 0, stream >> > (
				tensor->gpu<float>(n),
				tensor_width, tensor_height, stride, threshold, threshold_desigmoid, num_classes,
				anchor, bboxptr, counter, area, max_objs_, job_count);

			cudaMemcpyAsync(cpuPtrInput, gpuPtrInput, objsStoreSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);

			cpuPtrInput += objsStoreSize;
			gpuPtrInput += objsStoreSize;
		}
		cudaStreamSynchronize(stream);

		cpuPtrInput = (char*)cpuPtr;
		for (int n = 0; n < batchSize; ++n, cpuPtrInput += objsStoreSize) {
			auto& output = bboxs[n];

			int num = *((int*)cpuPtrInput);
			num = std::min(num, max_objs_);
			if (num == 0)
				continue;

			ccutil::BBox* ptr = (ccutil::BBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
	}
};