

#include "centerface_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	CenterFaceBackend::CenterFaceBackend(int max_objs, CUStream stream) :Backend(stream) {
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

	static __global__ void CenterFaceBackend_forwardGPU(float* hm, float* wh, float* offset, float* landmark, 
		int* countptr, ccutil::FaceBox* boxptr,
		int width, int height, int heatmapArea, int stride, float threshold,
		int maxobjs, int edge) {

		KERNEL_POSITION;

		float confidence = hm[position];
		if (confidence < threshold)
			return;

		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		int cx = position % width;
		int cy = position / width;

		ccutil::FaceBox* ptr = boxptr + index;
		float dh = commonExp(wh[position]) * stride;
		float dw = commonExp(wh[position + heatmapArea]) * stride;
		float oy = offset[position];
		float ox = offset[position + heatmapArea];

		ptr->x = (cx + ox + 0.5) * stride - dw * 0.5;
		ptr->y = (cy + oy + 0.5) * stride - dh * 0.5;
		ptr->r = ptr->x + dw;
		ptr->b = ptr->y + dh;
		ptr->score = confidence;
		ptr->label = 0;

		for (int k = 0; k < 5; ++k) {
			float landmark_x = ptr->x + landmark[heatmapArea * (k * 2 + 1) + position] * dw;
			float landmark_y = ptr->y + landmark[heatmapArea * (k * 2) + position] * dh;

			cv::Point2f& point = ptr->landmark[k];
			point.x = landmark_x;
			point.y = landmark_y;
		}
	}

	void CenterFaceBackend::postProcessGPU(shared_ptr<Tensor> heatmap, shared_ptr<Tensor> wh,
		shared_ptr<Tensor> offset, shared_ptr<Tensor> landmark,
		int stride, float threshold, vector<vector<ccutil::FaceBox>>& bboxs) {

		int batchSize = heatmap->num();
		int width = heatmap->width();
		int height = heatmap->height();
		int heatmapArea = width * height;
		int job_count = heatmapArea;	// c * h * w
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);
		size_t objsStoreSize = max_objs_ * sizeof(ccutil::FaceBox) + sizeof(int);
		
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();

		for (int i = 0; i < batchSize; ++i) {
			float* hm_ptr = heatmap->gpu<float>(i);
			float* wh_ptr = wh->gpu<float>(i);
			float* offset_ptr = offset->gpu<float>(i);
			float* landmark_ptr = landmark->gpu<float>(i);
			int* countPtr = (int*)gpuPtrInput;
			ccutil::FaceBox* boxPtr = (ccutil::FaceBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			CenterFaceBackend_forwardGPU <<< grid, block, 0, stream >>> (hm_ptr, wh_ptr, offset_ptr, landmark_ptr, 
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