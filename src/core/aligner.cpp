#include "aligner.h"


Aligner::Aligner() {}

Aligner::~Aligner() {}

int Aligner::AlignFace(const Mat &img_src, const vector<Point2f> &keypoints, Mat *face_aligned) {
	INFO("start align face.");
	if (img_src.empty()) {
		INFO("input empty.");
		return 10001;
	}
	if (keypoints.size() == 0) {
		INFO("keypoints empty.");
		return 10001;
	}	
	int num_keypoints = static_cast<int>(keypoints.size());
	vector<Point2f> points_src(5);
	switch (num_keypoints) {
	case 5:
		points_src = keypoints;
		break;
	case 98:
		points_src[0] = keypoints[96];
		points_src[1] = keypoints[97];
		points_src[2] = keypoints[54];
		points_src[3] = keypoints[76];
		points_src[4] = keypoints[82];
		break;
	case 106:
		points_src[0] = keypoints[104];
		points_src[1] = keypoints[105];
		points_src[2] = keypoints[46];
		points_src[3] = keypoints[84];
		points_src[4] = keypoints[90];
		break;
	default:
		INFOE("error keypoints num %d", num_keypoints);
		break;
	}

	Mat src_mat(5, 2, CV_32FC1, &points_src[0]);
	Mat dst_mat(5, 2, CV_32FC1, points_dst);
	Mat transform = SimilarTransform(src_mat, dst_mat);
	face_aligned->create(112, 112, CV_32FC3);
	Mat transfer_mat = transform(Rect(0, 0, 3, 2));
	warpAffine(img_src.clone(), *face_aligned, transfer_mat, Size(112, 112), 1, 0, 0);
	INFO("end align face.");

	return 0;
}

Mat Aligner::MeanAxis0(const Mat &src) {
	int num = src.rows;
	int dim = src.cols;

	// x1 y1
	// x2 y2
	Mat output(1, dim, CV_32FC1);
	for (int i = 0; i < dim; i++) {
		float sum = 0;
		for (int j = 0; j < num; j++) {
			sum += src.at<float>(j, i);
		}
		output.at<float>(0, i) = sum / num;
	}

	return output;
}

Mat Aligner::ElementwiseMinus(const Mat &A, const Mat &B) {
	Mat output(A.rows, A.cols, A.type());
	assert(B.cols == A.cols);
	if (B.cols == A.cols) {
		for (int i = 0; i < A.rows; i++) {
			for (int j = 0; j < B.cols; j++) {
				output.at<float>(i, j) = A.at<float>(i, j) - B.at<float>(0, j);
			}
		}
	}
	return output;	
}

Mat Aligner::VarAxis0(const Mat &src) {
	Mat temp_ = ElementwiseMinus(src, MeanAxis0(src));
	multiply(temp_, temp_, temp_);
	return MeanAxis0(temp_);
}

int Aligner::MatrixRank(const Mat &M) {
	Mat w, u, vt;
	SVD::compute(M.clone(), w, u, vt);
	Mat1b nonZeroSingularValues = w > 0.0001;
	int rank = countNonZero(nonZeroSingularValues);
	return rank;
}

/*
References: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
Anthor: Jack Yu
*/
Mat Aligner::SimilarTransform(const Mat &src, const Mat &dst) {
	int num = src.rows;
	int dim = src.cols;
	Mat src_mean = MeanAxis0(src);
	Mat dst_mean = MeanAxis0(dst);
	Mat src_demean = ElementwiseMinus(src, src_mean);
	Mat dst_demean = ElementwiseMinus(dst, dst_mean);
	Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
	Mat d(dim, 1, CV_32F);
	d.setTo(1.0f);
	if (determinant(A) < 0) {
		d.at<float>(dim - 1, 0) = -1;

	}
	Mat T = Mat::eye(dim + 1, dim + 1, CV_32F);
	Mat U, S, V;
	SVD::compute(A, S, U, V);

	// the SVD function in opencv differ from scipy .

	int rank = MatrixRank(A);
	if (rank == 0) {
		assert(rank == 0);

	} else if (rank == dim - 1) {
		if (determinant(U) * determinant(V) > 0) {
			T.rowRange(0, dim).colRange(0, dim) = U * V;
		} else {
			int s = d.at<float>(dim - 1, 0) = -1;
			d.at<float>(dim - 1, 0) = -1;

			T.rowRange(0, dim).colRange(0, dim) = U * V;
			Mat diag_ = Mat::diag(d);
			Mat twp = diag_ * V; //np.dot(np.diag(d), V.T)
			Mat B = Mat::zeros(3, 3, CV_8UC1);
			Mat C = B.diag(0);
			T.rowRange(0, dim).colRange(0, dim) = U * twp;
			d.at<float>(dim - 1, 0) = s;
		}
	} else {
		Mat diag_ = Mat::diag(d);
		Mat twp = diag_ * V.t(); //np.dot(np.diag(d), V.T)
		Mat res = U * twp; // U
		T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
	}
	Mat var_ = VarAxis0(src_demean);
	float val = sum(var_).val[0];
	Mat res;
	multiply(d, S, res);
	float scale = 1.0 / val * sum(res).val[0];
	T.rowRange(0, dim).colRange(0, dim) = -T.rowRange(0, dim).colRange(0, dim).t();
	Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
	Mat  temp2 = src_mean.t();
	Mat  temp3 = temp1 * temp2;
	Mat temp4 = scale * temp3;
	T.rowRange(0, dim).colRange(dim, dim + 1) = -(temp4 - dst_mean.t());
	T.rowRange(0, dim).colRange(0, dim) *= scale;
	return T;
}

