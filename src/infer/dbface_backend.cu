

#include "dbface_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	DBFaceBackend::DBFaceBackend(int max_objs, CUStream stream) :Backend(stream) {
		this->max_objs_ = max_objs;
	}

	static __device__ float commonExp(float value) {

		float gate = 1.0f;
		if (fabs(value) < gate)
			return value * exp(gate);

		if (value > 0)
			return exp(value);
		else
			return -exp(-value);
	}

	static __global__ void DBFaceBackend_forwardGPU(float* hm, float* hmpool, float* tlrb, float* landmark,
		int* countptr, ccutil::FaceBox* boxptr,
		int width, int height, int heatmapArea, int stride, float threshold,
		int maxobjs, int edge) {

		KERNEL_POSITION;

		float confidence = hm[position];
		if (confidence != hmpool[position] || confidence < threshold)
			return;

		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		int cx = position % width;
		int cy = position / width;
		int oc0 = position;
		int oc1 = position + heatmapArea;
		int oc2 = position + heatmapArea * 2;
		int oc3 = position + heatmapArea * 3;

		ccutil::FaceBox* ptr = boxptr + index;
		float dx = tlrb[oc0];
		float dy = tlrb[oc1];
		float dr = tlrb[oc2];
		float db = tlrb[oc3];
		ptr->x = (cx - dx) * stride;
		ptr->y = (cy - dy) * stride;
		ptr->r = (cx + dr) * stride;
		ptr->b = (cy + db) * stride;
		ptr->score = confidence;
		ptr->label = 0;

		for (int k = 0; k < 5; ++k) {
			float landmark_x = landmark[position + heatmapArea * k] * 4;
			float landmark_y = landmark[position + heatmapArea * (k + 5)] * 4;

			cv::Point2f& point = ptr->landmark[k];
			point.x = (commonExp(landmark_x) + cx) * stride;
			point.y = (commonExp(landmark_y) + cy) * stride;
		}
	}

	void DBFaceBackend::postProcessGPU(shared_ptr<Tensor> outHM, shared_ptr<Tensor> outHMPool,
		shared_ptr<Tensor> outTLRB, shared_ptr<Tensor> outLandmark,
		int stride, float threshold, vector<vector<ccutil::FaceBox>>& bboxs) {

		int batchSize = outHM->num();
		int width = outHM->width();
		int height = outHM->height();
		int heatmapArea = width * height;
		int job_count = outHM->count(1);	// c * h * w
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);
		size_t objsStoreSize = max_objs_ * sizeof(ccutil::FaceBox) + sizeof(int);
		
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();

		for (int i = 0; i < batchSize; ++i) {
			float* hm_ptr = outHM->gpu<float>(i);
			float* hmpool_ptr = outHMPool->gpu<float>(i);
			float* tlrb_ptr = outTLRB->gpu<float>(i);
			float* landmark_ptr = outLandmark->gpu<float>(i);
			int* countPtr = (int*)gpuPtrInput;
			ccutil::FaceBox* boxPtr = (ccutil::FaceBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			DBFaceBackend_forwardGPU <<< grid, block, 0, stream >>> (hm_ptr, hmpool_ptr, tlrb_ptr, landmark_ptr,
				countPtr, boxPtr, width, height, heatmapArea, stride, threshold, max_objs_, job_count);

			cudaMemcpyAsync(cpuPtrInput, gpuPtrInput, objsStoreSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream);
			cpuPtrInput += objsStoreSize;
			gpuPtrInput += objsStoreSize;
		}
		cudaStreamSynchronize(stream);

		cpuPtrInput = (char*)cpuPtr;
		for (int n = 0; n < batchSize; ++n, cpuPtrInput += objsStoreSize) {
			auto& output = bboxs[n];
			output.clear();
			int num = *((int*)cpuPtrInput);
			num = std::min(num, max_objs_);
			if (num == 0) 
				continue;

			ccutil::FaceBox* ptr = (ccutil::FaceBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
	}
};