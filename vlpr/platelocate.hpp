#ifndef VLPR_CORE_PLATELOCATE_H_
#define VLPR_CORE_PLATELOCATE_H_

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/features2d/features2d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include <iostream> 
#include <string>
#include "basic_class.hpp"
#include "config.h"

//图片标准大小1600*1264
//#include"config.h"

#define DEBUG

using namespace cv;
using namespace std;

namespace Locate{
	void findRect(const Mat &clear_gray, const Mat &binary_mat, vector<Mat> &out);
	void verifyPointsNum(vector<vector<Point>> contours, int &low, int &up);
    bool verifySizes(RotatedRect mr, const float& min, const float &max);

	bool calcSafeRect(const RotatedRect &roi_rect, const Mat &src,
		Rect_<float> &safeBoundRect);
	bool calcSafeBigRect(const RotatedRect &roi_rect, const Mat &src,
		Rect_<float> &safeBoundRect);
	bool rotation(Mat &in, Mat &out, const Size rect_size,
		const Point2f center, const double angle);
	Mat deflection(const Mat &in_b, const Mat &in);
	void cross_rectMerge(vector<RotatedRect> &rects, float verifyMin, float verifyMax);
	Mat scharrOper(const Mat& src,double ratio);

	//void  blue_normal_morph(const Mat& in, vector<Mat> &out); 
	//void yellow_colorMorph(const Mat &in, vector< Mat>& out); 
	Mat  blue_normal_morph(const Mat& in); 
	Mat yellow_colorMorph(const Mat &in); 
	Mat sobelMorph(const Mat& in);

	void threshRegion(const Mat& mat_gray, Mat &thresh, int x, int y, bool inv); 
	Mat maxGradient(const Mat& in);
	Mat tooLightP_morph(const Mat& in);
    Mat darkP_morph(const Mat& in); 
	bool findCPlate(const cv::Mat &clear_gray, const cv::Mat &binary_mat, std::vector<CPlate> &out);
	void  plateDetect(const Mat &in, vector<Mat> &out);
	void  CplateDetect(const Mat &in, vector<CPlate> &out, const std::vector<bool> &methods);
	}

#endif