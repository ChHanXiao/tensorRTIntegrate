

#include "centernet_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	CenterNetBackend::CenterNetBackend(int max_objs, CUStream stream) :Backend(stream) {
		this->max_objs_ = max_objs;
	}

	static __global__ void CenterNetBackend_forwardGPU(float* hm, float* hmpool, float* wh, float* offset,
		int* countptr, ccutil::BBox* boxptr,
		int width, int height, int heatmapArea, int stride, float threshold,
		int maxobjs, int edge) {

		KERNEL_POSITION;

		float confidence = hm[position];
		if (confidence != hmpool[position] || confidence < threshold)
			return;

		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		int channel_index = position / heatmapArea;
		int classes = channel_index;
		int offsetChannel0 = position - channel_index * heatmapArea;
		int offsetChannel1 = offsetChannel0 + heatmapArea;

		int cx = offsetChannel0 % width;
		int cy = offsetChannel0 / width;

		ccutil::BBox* ptr = boxptr + index;
		float dw = (wh[offsetChannel0]) * stride;
		float dh = wh[offsetChannel1] * stride;
		float ox = offset[offsetChannel0];
		float oy = offset[offsetChannel1];
		ptr->x = (cx + ox + 0.5) * stride - dw * 0.5;
		ptr->y = (cy + oy + 0.5) * stride - dh * 0.5;
		ptr->r = ptr->x + dw;
		ptr->b = ptr->y + dh;
		ptr->score = confidence;
		ptr->label = classes;
	}

	void CenterNetBackend::postProcessGPU(shared_ptr<Tensor> outHM, shared_ptr<Tensor> outHMPool,
		shared_ptr<Tensor> outWH, shared_ptr<Tensor> outOffset,
		int stride, float threshold, vector<vector<ccutil::BBox>>& bboxs) {

		int batchSize = outHM->num();
		int width = outHM->width();
		int height = outHM->height();
		int heatmapArea = width * height;
		int job_count = outHM->count(1);	// c * h * w
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);
		size_t objsStoreSize = max_objs_ * sizeof(ccutil::BBox) + sizeof(int);
		
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();

		for (int i = 0; i < batchSize; ++i) {
			float* hm_ptr = outHM->gpu<float>(i);
			float* hmpool_ptr = outHMPool->gpu<float>(i);
			float* wh_ptr = outWH->gpu<float>(i);
			float* offset_ptr = outOffset->gpu<float>(i);

			int* countPtr = (int*)gpuPtrInput;
			ccutil::BBox* boxPtr = (ccutil::BBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			CenterNetBackend_forwardGPU <<< grid, block, 0, stream >>> (hm_ptr, hmpool_ptr, wh_ptr, offset_ptr,
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

			ccutil::BBox* ptr = (ccutil::BBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
	}
};