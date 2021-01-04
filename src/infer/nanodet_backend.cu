

#include "nanodet_backend.hpp"
#include <cc_util.hpp>
#include <common/trt_common.hpp>

namespace TRTInfer {

	NanoDetBackend::NanoDetBackend(int max_objs, CUStream stream) :Backend(stream) {

		this->max_objs_ = max_objs;
	}

	static __device__ float sigmoid(float value) {
		return 1 / (1 + exp(-value));
	}

	static float desigmoid(float val) {
		return -log(1 / val - 1);
	}


	static __global__ void decode_native_impl(float* cls, float* loc,
		int width, int height, int stride, float threshold, int num_classes,
		ccutil::BBox* output, int* counter, int maxobjs, int reg_max, int edge) {

		KERNEL_POSITION;

		float* pclasses = cls + position * num_classes;
		float max_class_confidence = *pclasses;
		int max_classes = 0;
		for (int j = 1; j < num_classes; ++j, ++pclasses) {
			if (*pclasses > max_class_confidence) {
				max_classes = j;
				max_class_confidence = *pclasses;
			}
		}
		if (max_class_confidence < threshold)
			return;
		int index = atomicAdd(counter, 1);
		if (index >= maxobjs)
			return;
		int length = reg_max + 1;
		float* plocation = loc + position * 4 * length;
		int celly = position / width;
		int cellx = position % width;
		float ct_x = (cellx + 0.5) * stride;
		float ct_y = (celly + 0.5) * stride;
		float* dis_pred = new float[4];
		for (int i = 0; i < 4; i++) {
			float* ptr = plocation + i * length;
			float* ptrmax = ptr;
			float alpha = *ptrmax;
			for (int j = 1; j < length; ++j, ++ptrmax) {
				if (*ptrmax > alpha) {
					alpha = *ptrmax;
				}
			}
			float denominator = 0;
			float dis = 0;
			float* dis_after_sm = new float[length];
			for (int j = 0; j < length; ++j) {
				dis_after_sm[j] = exp(ptr[j] - alpha);
				denominator += dis_after_sm[j];
			}
			for (int j = 0; j < length; ++j) {
				dis_after_sm[j] /= denominator;
			}
			for (int j = 0; j < length; j++)
			{
				dis += j * dis_after_sm[j];
			}
			dis *= stride;
			dis_pred[i] = dis;
			delete[] dis_after_sm;
		}

		float x = (ct_x - dis_pred[0]);
		float y = (ct_y - dis_pred[1]);
		float r = (ct_x + dis_pred[2]);
		float b = (ct_y + dis_pred[3]);
		delete[] dis_pred;
		ccutil::BBox& box = output[index];
		box.x = x;
		box.y = y;
		box.r = r;
		box.b = b;
		box.label = max_classes;
		box.score = max_class_confidence;

	}


	void NanoDetBackend::postProcessGPU(shared_ptr<Tensor> cls_tensor, shared_ptr<Tensor> loc_tensor, int stride,
		Size netInputSize, float threshold, int num_classes, vector<vector<ccutil::BBox>> &bboxs, int reg_max_){
		
		int batchSize = cls_tensor->num();
		int tensor_channel = cls_tensor->channel();
		int feature_area = cls_tensor->height();

		int feature_h = netInputSize.height / stride;
		int feature_w = netInputSize.width / stride;
		//float threshold_desigmoid = desigmoid(threshold);

		size_t objsStoreSize = max_objs_ * sizeof(ccutil::BBox) + sizeof(int);
		void* cpuPtr = getCPUMemory(objsStoreSize * batchSize);
		char* cpuPtrInput = (char*)cpuPtr;
		void* gpuPtr = getGPUMemory(objsStoreSize * batchSize);
		char* gpuPtrInput = (char*)gpuPtr;
		auto stream = getStream();
		int job_count = feature_area;
		auto grid = gridDims(job_count);
		auto block = blockDims(job_count);

		for (int n = 0; n < batchSize; ++n) {

			int* counter = (int*)gpuPtrInput;
			ccutil::BBox* bboxptr = (ccutil::BBox*)((char*)gpuPtrInput + sizeof(int));

			cudaMemsetAsync(counter, 0, sizeof(int), stream);
			decode_native_impl << < grid, block, 0, stream >> > (
				cls_tensor->gpu<float>(n), loc_tensor->gpu<float>(n), feature_w, feature_h,
				stride, threshold, num_classes, bboxptr, counter, max_objs_, reg_max_, job_count);

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