

#include "retinalp_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	RetinaLPBackend::RetinaLPBackend(int max_objs, CUStream stream) :Backend(stream) {
		this->max_objs_ = max_objs;
	}

	static __global__ void RetinaLPBackend_forwardGPU(float* conf, float* offset, float* landmark, float *anchors_matrix,
		int* countptr, ccutil::LPRBox* boxptr, float threshold, int maxobjs, int edge) {

		KERNEL_POSITION;

		float confidence = conf[position * 2 + 1];
		if (confidence < threshold)
			return;
		int index = atomicAdd(countptr, 1);
		if (index >= maxobjs)
			return;

		float cx_a = anchors_matrix[position * 4];
		float cy_a = anchors_matrix[position * 4 + 1];
		float w_a = anchors_matrix[position * 4 + 2];
		float h_a = anchors_matrix[position * 4 + 3];
		//printf("anchors_matrix:%f,%f,%f,%f \n", cx_a, cy_a, w_a, h_a);
		float loc_x = offset[position * 4];
		float loc_y = offset[position * 4 + 1];
		float loc_w = offset[position * 4 + 2];
		float loc_h = offset[position * 4 + 3];
		float cx_b = cx_a + loc_x * 0.1 * w_a;
		float cy_b = cy_a + loc_y * 0.1 * h_a;
		float w_b = w_a * expf(loc_w * 0.2);
		float h_b = h_a * expf(loc_h * 0.2);

		ccutil::LPRBox* ptr = boxptr + index;
		ptr->x = cx_b - w_b * 0.5;
		ptr->y = cy_b - h_b * 0.5;
		ptr->r = cx_b + w_b * 0.5;
		ptr->b = cy_b + h_b * 0.5;
		ptr->score = confidence;
		ptr->label = 0;

		for (int k = 0; k < 4; ++k) {
			float landmark_x = cx_a + 0.1 * landmark[position *8 + k * 2] * w_a;
			float landmark_y = cy_a + 0.1 * landmark[position *8 + k * 2 + 1] * h_a;

			cv::Point2f& point = ptr->landmark[k];
			point.x = landmark_x;
			point.y = landmark_y;
		}
	}

	void RetinaLPBackend::postProcessGPU(shared_ptr<Tensor> conf, shared_ptr<Tensor> offset,
		shared_ptr<Tensor> landmark, Mat anchors_matrix, 
		float threshold, vector<vector<ccutil::LPRBox>> &bboxs) {

		int batchSize = conf->num();
		int total_pix = conf->height();
		int job_count = total_pix;
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);
		size_t objsStoreSize = max_objs_ * sizeof(ccutil::LPRBox) + sizeof(int);

		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;

		void* gpu_anchors_Ptr = getGPUMemory(total_pix * 4 * sizeof(float));
		float* gpu_anchors_matrix_ = (float*)gpu_anchors_Ptr;

		auto stream = getStream();
		float* anchors_matrix_ptr = anchors_matrix.ptr<float>(0);
		cudaMemcpyAsync(gpu_anchors_matrix_, anchors_matrix_ptr, total_pix * 4 * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice, stream);

		for (int i = 0; i < batchSize; ++i) {

			float* conf_ptr = conf->gpu<float>(i);
			float* offset_ptr = offset->gpu<float>(i);
			float* landmark_ptr = landmark->gpu<float>(i);
			int* countPtr = (int*)gpuPtrInput;
			ccutil::LPRBox* boxPtr = (ccutil::LPRBox*)((char*)gpuPtrInput + sizeof(int));
			cudaMemsetAsync(gpuPtrInput, 0, sizeof(int), stream);
			RetinaLPBackend_forwardGPU << < grid, block, 0, stream >> > (conf_ptr, offset_ptr, landmark_ptr, gpu_anchors_matrix_,
				countPtr, boxPtr, threshold, max_objs_, job_count);

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

			ccutil::LPRBox* ptr = (ccutil::LPRBox*)(cpuPtrInput + sizeof(int));
			output.insert(output.begin(), ptr, ptr + num);
		}
	}
};