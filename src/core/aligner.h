#pragma once

#ifndef ALIGNER_H
#define ALIGNER_H

#include "trtmodel.h"

class Aligner {
public:
	Aligner();
	~Aligner();

	int AlignFace(const Mat &img_src, const vector<Point2f>& keypoints, Mat *face_aligned);
	Mat MeanAxis0(const Mat &src);
	Mat ElementwiseMinus(const Mat &A, const Mat &B);
	Mat VarAxis0(const Mat &src);
	int MatrixRank(const Mat &M);
	Mat SimilarTransform(const Mat &src, const Mat &dst);

private:

	float points_dst[5][2] = {
		{ 30.2946f + 8.0f, 51.6963f },
		{ 65.5318f + 8.0f, 51.5014f },
		{ 48.0252f + 8.0f, 71.7366f },
		{ 33.5493f + 8.0f, 92.3655f },
		{ 62.7299f + 8.0f, 92.2041f }
	};

};


#endif

