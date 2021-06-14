#include "crop_lp.h"

int crop_lp(const cv::Mat& img_src, const ccutil::LPRBox lpbox, cv::Mat* img_crop) {
	int xmin = int(std::max(lpbox.x, 0.f));
	int ymin = int(std::max(lpbox.y, 0.f));
	int xmax = int(std::min(lpbox.r, (float)img_src.cols));
	int ymax = int(std::min(lpbox.b, (float)img_src.rows));
	cv::Rect rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
	cv::Mat plate_src = img_src(rect);
	cv::Point2f srcPts[4];

	srcPts[0].x = lpbox.landmark[2].x - xmin;
	srcPts[0].y = lpbox.landmark[2].y - ymin;
	srcPts[1].x = lpbox.landmark[3].x - xmin;
	srcPts[1].y = lpbox.landmark[3].y - ymin;
	srcPts[2].x = lpbox.landmark[1].x - xmin;
	srcPts[2].y = lpbox.landmark[1].y - ymin;
	srcPts[3].x = lpbox.landmark[0].x - xmin;
	srcPts[3].y = lpbox.landmark[0].y - ymin;

	cv::Mat perspectiveMat = getPerspectiveTransform(srcPts, points_ref);
	warpPerspective(plate_src, *img_crop, perspectiveMat, plate_size);
	return 0;
}
