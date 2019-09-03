#ifndef VLPR_CORE_PLATELOCATE_CPP_
#define VLPR_CORE_PLATELOCATE_CPP_

#include"platelocate.hpp"

//图片标准大小1600*1264
//#include"config.h"

//#define DEBUG

using namespace cv;
using namespace std;

namespace Locate{

	void verifyPointsNum(vector<vector<Point>> contours, int &low, int &up){
		vector<vector<Point>>::iterator itc = contours.begin();
		vector<int> pointnum;
		while (itc != contours.end()) {
			int a = (*itc).size();
			pointnum.push_back(a);
			itc++;
		}

		sort(pointnum.begin(), pointnum.end());
		vector<int>::iterator upitc, lowitc;
		upitc = pointnum.end() - 1;
		if (pointnum.size() > 30)
			lowitc = pointnum.end() - 30;
		else
			lowitc = pointnum.begin();
		up = *upitc;
		low = *lowitc;
		return;
	}

	bool verifySizes(RotatedRect mr, const float& min, const float &max) {
		// China car plate size: 440mm*140mm��aspect 3.142857
		float rmin = 1.3;//1.27
		float rmax = 6.0;//4.71
		float mr_angle = mr.angle;

		float area = mr.size.height * mr.size.width;
		float r = (float)mr.size.width / (float)mr.size.height;
		if (r < 1){
			r = (float)mr.size.height / (float)mr.size.width;
			mr_angle = -(90 + mr_angle);
		}
		if ((area > min) && (area < max) && (r > rmin) && (mr_angle + 45>0) && (r < rmax))
			//cout << area << endl;
			return true;
		else
			return false;
	}

	bool calcSafeRect(const RotatedRect &roi_rect, const Mat &src,
		Rect_<float> &safeBoundRect) {
		Rect_<float> boudRect = roi_rect.boundingRect();

		float tl_x = boudRect.x > 0 ? boudRect.x : 0;
		float tl_y = boudRect.y > 0 ? boudRect.y : 0;

		float br_x = (boudRect.x + boudRect.width)< src.cols ?
			(boudRect.x + boudRect.width - 1)
			: src.cols - 1;
		float br_y = (boudRect.y + boudRect.height)< src.rows ?
			(boudRect.y + boudRect.height - 1)
			: src.rows - 1;

		float roi_width = br_x - tl_x;
		float roi_height = br_y - tl_y;

		if (roi_width <= 0 || roi_height <= 0) return false;

		//  a new rect not out the range of mat

		safeBoundRect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);

		return true;
	}
	bool calcSafeBigRect(const RotatedRect &roi_rect, const Mat &src,
		Rect_<float> &safeBoundRect) {
		Rect_<float> boudRect = roi_rect.boundingRect();
		float tlx = boudRect.x - boudRect.width*0.2;
		float tly = boudRect.y - boudRect.height*0.2;
		float bw = boudRect.width*1.4;
		float bh = boudRect.height*1.4;
		float tl_x = tlx > 0 ? tlx : 0;
		float tl_y = tly> 0 ? tly : 0;

		float br_x = (tlx + bw)< src.cols ? (tlx + bw - 1) : (src.cols - 1);
		float br_y = (tly + bh)< src.rows ? (tly + bh - 1) : (src.rows - 1);

		float roi_width = br_x - tl_x;
		float roi_height = br_y - tl_y;

		if (roi_width <= 0 || roi_height <= 0) return false;

		//  a new rect not out the range of mat
		safeBoundRect = Rect_<float>(tl_x, tl_y, roi_width, roi_height);
		return true;
	}

	bool rotation(Mat &in, Mat &out, const Size rect_size,
		const Point2f center, const double angle) {

		Mat in_large(int(in.rows * 1.5), int(in.cols * 1.5), CV_8UC1, Scalar(0));

		float x = in_large.cols / 2 - center.x > 0 ? in_large.cols / 2 - center.x : 0;
		float y = in_large.rows / 2 - center.y > 0 ? in_large.rows / 2 - center.y : 0;

		float width = x + in.cols < in_large.cols ? in.cols : in_large.cols - x;
		float height = y + in.rows < in_large.rows ? in.rows : in_large.rows - y;

		if (width != in.cols || height != in.rows) { return false; }

		Mat imageRoi = in_large(Rect_<float>(x, y, width, height));
		in.copyTo(imageRoi);

		Point2f new_center(x + center.x, y + center.y);

		Mat rot_mat = getRotationMatrix2D(new_center, angle, 1);

		Mat mat_rotated;
		warpAffine(in_large, mat_rotated, rot_mat, Size(in_large.cols, in_large.rows),
			CV_INTER_CUBIC);

		getRectSubPix(mat_rotated, rect_size, new_center, out);

		return true;
	}

	Mat deflection(const Mat &in_b, const Mat &in)
	{
		int nRows = in_b.rows;
		int nCols = in_b.cols;
		//assert(in_b.channels() == 3);

		vector<Point> points1;
		vector<Point> points2;
		double slope1, slope2;
		double slope;
		const uchar* p;

		for (int i = 0; i < nRows; i++){
			p = in_b.ptr<uchar>(i);
			int j = 0, k = nCols;
			int value = int(p[0]);
			int valueend = 0;
			while (0 == value && j < (nCols - 1)) value = int(p[++j]);
			while (0 == valueend &&  k> 0)      valueend = int(p[--k]);
			points1.push_back(Point(j, i));
			points2.push_back(Point(k, i));
		}
		//cout << points1 << endl;
		//cout << points2 << endl;

		Vec4f line1;
		Vec4f line2;
		fitLine(points1, line1, CV_DIST_HUBER, 0, 0.01, 0.01);
		fitLine(points2, line2, CV_DIST_HUBER, 0, 0.01, 0.01);
		//cout << line1 << endl;
		//cout << line2 << endl;

		if (abs(line1[0]) > 0.1 && abs(line2[0]) > 0.1)//line[0]=cosQ, line[1]=sinQ,
		{
			slope1 = line1[0] / line1[1];
			slope2 = line2[0] / line2[1];
			if (slope1 > 0 && slope2 > 0)//right bias
				slope = min(slope1, slope2);//(slope1+slope2)/2
			else if (slope1 < 0 && slope2 < 0)//left bias
				slope = max(slope1, slope2);
			else 
				return in;
			Point2f dstTri[3];
			Point2f plTri[3];

			float height = (float)in.rows;
			float width = (float)in.cols;
			float xiff = (float)abs(slope) * height;

			if (slope >0) {
				plTri[0] = Point2f(0, 0);
				plTri[1] = Point2f(width - xiff - 1, 0);
				plTri[2] = Point2f(0 + xiff, height - 1);

				dstTri[0] = Point2f(xiff / 2, 0);
				dstTri[1] = Point2f(width - 1 - xiff / 2, 0);
				dstTri[2] = Point2f(xiff / 2, height - 1);
			}
			else {
				plTri[0] = Point2f(0 + xiff, 0);
				plTri[1] = Point2f(width - 1, 0);
				plTri[2] = Point2f(0, height - 1);

				dstTri[0] = Point2f(xiff / 2, 0);
				dstTri[1] = Point2f(width - 1 - xiff + xiff / 2, 0);
				dstTri[2] = Point2f(xiff / 2, height - 1);
			}

			Mat warp_mat = getAffineTransform(plTri, dstTri);

			Mat affine_mat;
			affine_mat.create((int)height, (int)width, CV_8UC1);

			if (in.rows > 36 || in.cols > 136)
				warpAffine(in, affine_mat, warp_mat, affine_mat.size(), CV_INTER_AREA);
			else
				warpAffine(in, affine_mat, warp_mat, affine_mat.size(), CV_INTER_CUBIC);
			//imwrite("./temp/img8/affine.jpg", affine_mat);
			return affine_mat;
		}
		else{
			slope = 0;//cot(85)=0.087
			return in;
		}
	}

	void cross_rectMerge(vector<RotatedRect> &rects,float verifyMin,float verifyMax){
		size_t sizeR = rects.size();
		//cout << sizeR << endl;
		for (size_t i = 0; i < sizeR- 1; i++){
			Point2f i_points[4];
			rects[i].points(i_points);
			int areai = rects[i].size.width*rects[i].size.height;
			int H = min(min(min(i_points[1].y, i_points[2].y), i_points[3].y), i_points[0].y);
			int B = max(max(max(i_points[1].y, i_points[2].y), i_points[3].y), i_points[0].y);
			int L = min(min(min(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			int R = max(max(max(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			for (size_t j = i + 1; j < sizeR; j++){
				Point2f j_points[4];
				rects[j].points(j_points);
				int k = 0;
				while (k < 4){
					if ((j_points[k].x<R) && (j_points[k].x>L) && (j_points[k].y>H) && (j_points[k].y < B)){
						int areaj = rects[j].size.width*rects[j].size.height;
						float ratio = (float)areai / (float)areaj;
						if ((0.33 < ratio) && (ratio < 3)){
							vector<Point> mixPoints = { j_points[0], j_points[1], j_points[2], j_points[3], i_points[0], i_points[1], i_points[2], i_points[3] };
							RotatedRect mixed = minAreaRect(mixPoints);
							if (verifySizes(mixed, verifyMin, verifyMax))
							   rects.push_back(mixed);
						}
						break;
					}
					++k;
				}
			}
		}
	}

	
	Mat scharrOper(const Mat& src,double ratio_y){
		Mat scharr = Mat::zeros(src.size(), src.type());
		if (src.channels() != 1){
			cout << "Wrong picture channel in ScharrOper" << endl;
			return scharr;
		}
		Mat grad_x, grad_y;
		Mat temp(scharr.size(), CV_32FC1, Scalar(0));
		Scharr(src, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
		Scharr(src, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
		MatIterator_<short>  xi = grad_x.begin<short>();
		MatIterator_<short>   xj = grad_x.end<short>();
		MatIterator_<short>  yi = grad_y.begin<short>();
		MatIterator_<float>   s = temp.begin<float>();
		for (; xi != xj; ++xi, ++yi, ++s)
			*s = sqrt((*xi)*(*xi) +ratio_y*ratio_y*(*yi)*(*yi));//0.5
		normalize(temp, temp, 1, 0, CV_MINMAX,-1);
		temp.convertTo(scharr, CV_8UC1, 255, 0);
		return scharr;

	}

	Mat  blue_normal_morph(const Mat& in){
	
		vector<Mat> channels;
		split(in, channels);
		Mat blue = channels[0];
		Mat green = channels[1];
		Mat red = channels[2];
		Mat Eblue = scharrOper(blue,0.5);
		Mat Egreen = scharrOper(green,0.5);
		Mat  Ered = scharrOper(red,0.5);
		Mat Ibw = Mat::zeros(blue.size(), blue.type());
		MatIterator_<uchar> bw = Ibw.begin<uchar>();

		Mat Ibw_match;
		MatIterator_<uchar> b = blue.begin<uchar>();
		MatIterator_<uchar> g = green.begin<uchar>();
		MatIterator_<uchar> r = red.begin<uchar>();
		MatIterator_<uchar> eb = Eblue.begin<uchar>();
		MatIterator_<uchar> eg = Egreen.begin<uchar>();
		MatIterator_<uchar> er = Ered.begin<uchar>();
		MatIterator_<uchar> end = blue.end<uchar>();
		float alfa = 0.95, beta = 1.1;//0.95 1.1
		for (; b != end; ++b, ++g, ++r, ++eb, ++eg, ++er, ++bw){
			if ((*eb)<(alfa*(*er)) && (*eb)<(alfa*(*eg))
				&& (*b) > (beta*(*g)) && (*b) > (beta*(*r)))
				*bw = max(*eg, *er);
		}
		blur(Ibw, Ibw, Size(3, 3));
		threshold(Ibw, Ibw_match, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);

		Mat mat_morph, element;
		element = getStructuringElement(MORPH_RECT, Size(3, 3));//30 10 6 2
		morphologyEx(Ibw_match, mat_morph, MORPH_CLOSE, element);
		medianBlur(mat_morph, mat_morph, 5);
		element = getStructuringElement(MORPH_RECT, Size(40, 10));//30 10 6 2
		morphologyEx(Ibw_match, mat_morph, MORPH_CLOSE, element);
		medianBlur(mat_morph, mat_morph, 9);
		element = getStructuringElement(MORPH_RECT, Size(10,10));
		morphologyEx(mat_morph,mat_morph, MORPH_DILATE, element);
		return mat_morph;
		/*Mat Iyb = Mat::zeros(blue.size(), blue.type());
		MatIterator_<uchar> yb = Iyb.begin<uchar>();
		if ((*eg) < alfa*(*er) && (*eb)<alfa*(*eg) && (*b<60) && (*r) >(beta*(*b)) && (*r) >(beta*(*g)))
		*yb = *er;
		blur(Iyb, Iyb, Size(3, 3));
		threshold(Iyb, Iyb_match, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		addWeighted(Ibw_match, 1.0, Iyb_match, 1.0, 0.0, Ibw_match);*/
	}

	Mat yellow_colorMorph(const Mat &in){
		Mat mat_morph;
		Mat src_hsv;
		cvtColor(in, src_hsv, CV_BGR2HSV);
		vector<Mat> hsvSplit;
		int channels = src_hsv.channels();
		int nRows = src_hsv.rows;

		int nCols = src_hsv.cols * channels;
		if (src_hsv.isContinuous()) {
			nCols *= nRows;
			nRows = 1;
		}

		int i, j;
		uchar* p;
		float count = 0;
		for (i = 0; i < nRows; ++i) {
			p = src_hsv.ptr<uchar>(i);
			for (j = 0; j < nCols; j += 3) {
				int H = int(p[j]);      // 0-180
				float S = float(p[j + 1]) / 255.0;  // 0-255
				float L = float(p[j + 2]) / 255.0;  // 0-255
				count++;
				if (H > 10 && H < 32 && L>0.3&&L - S*S + 3 * S - 1.7 >0)
					//((S >= 64) )//&& (!(S<51) && (V>191)))
					p[j + 2] = 255;
				else
					p[j + 2] = 0;
			}
		}
		std::vector<cv::Mat> hsvSplit_done;
		split(src_hsv, hsvSplit_done);
		Mat mat_match = hsvSplit_done[2];
		medianBlur(mat_match, mat_match, 9);
		Mat element = getStructuringElement(MORPH_RECT, Size(30, 30));
		morphologyEx(mat_match, mat_morph, MORPH_CLOSE, element);
		//medianBlur(mat_match, mat_match, 5);
		//element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//morphologyEx(mat_match, mat_morph, MORPH_OPEN, element);
	
		return mat_morph;
	}

	Mat sobelMorph(const Mat& in){
		Mat mat_blur;
		GaussianBlur(in, mat_blur, Size(5, 5), 0, BORDER_DEFAULT);
		Mat mat_gray;
		cvtColor(mat_blur, mat_gray, CV_RGB2GRAY);

		Mat grad_x;
		Mat grad;
		Sobel(mat_gray, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		convertScaleAbs(grad_x, grad);

		Mat mat_threshold;
		threshold(grad, mat_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
		Mat mat_morph;
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));//27 9
		morphologyEx(mat_threshold, mat_morph, MORPH_CLOSE, element);
		medianBlur(mat_morph, mat_morph, 11);
		element = getStructuringElement(MORPH_RECT, Size(27, 1));//27 9
		morphologyEx(mat_threshold, mat_morph, MORPH_CLOSE, element);
		medianBlur(mat_morph, mat_morph, 7);
		return mat_morph;
	}

	void threshRegion(const Mat& mat_gray, Mat &thresh, int x, int y, bool inv){

		int time1 = mat_gray.cols / x;
		int time2 = mat_gray.rows / y;
		int xleft = mat_gray.cols - time1*x;
		int yleft = mat_gray.rows - time2*y;
		thresh = mat_gray.clone();
		int a = inv ? 1 : 0;
		//cout << time1 << "  " << time2 << "  " << xleft << "   " << yleft << endl;
		float ratio = 0.2;//0.15
		double value = 10;
		double he;
		float areas = mat_gray.cols*mat_gray.rows / (x*y*1.0);
		Mat car_threshroi, car_threshroi1, car_threshroi2, car_threshroi3;
		for (int i = 0; i <(x + 1); i++){
			if (i < x){
				for (int j = 0; j < (y + 1); j++){
					if (j < y){
						car_threshroi = thresh(Rect(i*time1, j*time2, time1, time2));
						he = threshold(car_threshroi, car_threshroi, 0, 255, CV_THRESH_OTSU + a);
						if (countNonZero(car_threshroi)>areas*ratio||he<value)
							car_threshroi.setTo(Scalar(0));
					}
					else if (yleft > 0){
						car_threshroi1 = thresh(Rect(i*time1, j*time2, time1, yleft));
						he = threshold(car_threshroi1, car_threshroi1, 0, 255, CV_THRESH_OTSU + a);
						if (countNonZero(car_threshroi1)>areas*ratio||he<value)
							car_threshroi1.setTo(Scalar(0));
					}

				}
			}
			else if (xleft != 0) {
				for (int j = 0; j < (y + 1); j++){
					if (j < y){
						car_threshroi2 = thresh(Rect(i*time1, j*time2, xleft, time2));
						he = threshold(car_threshroi2, car_threshroi2, 0, 255, CV_THRESH_OTSU + a);
						if (countNonZero(car_threshroi2)>areas*ratio||he<value)
							car_threshroi2.setTo(Scalar(0));
					}
					else if (yleft != 0){
						car_threshroi3 = thresh(Rect(i*time1, j*time2, xleft, yleft));
						he = threshold(car_threshroi3, car_threshroi3, 0, 255, CV_THRESH_OTSU + a);
						if (countNonZero(car_threshroi3)>areas*ratio||he<value)
							car_threshroi3.setTo(Scalar(0));
					}

				}
			}

		}
	}
	Mat maxGradient(const Mat& in){
		vector<Mat> channels;
		vector<Mat> out;
		out.reserve(3);
		out.push_back(in);
		split(in, channels);
		Mat blue = channels[0];
		Mat green = channels[1];
		Mat red = channels[2];
		//green.convertTo(green,CV_16S); 
		//red.convertTo(red, CV_16S);
		//convertScaleAbs(green*0.5 + red*0.5,sum);

		Mat Eblue = scharrOper(blue,0.1);
		Mat Egreen = scharrOper(green,0.1);
		Mat  Ered = scharrOper(red,0.1);
		//Mat Egray = scharrOper(gray);
		Mat Ibw = Mat::zeros(blue.size(), blue.type());
		MatIterator_<uchar> bw = Ibw.begin<uchar>();
		MatIterator_<uchar> eb = Eblue.begin<uchar>();
		MatIterator_<uchar> eg = Egreen.begin<uchar>();
		MatIterator_<uchar> er = Ered.begin<uchar>();
		MatIterator_<uchar> end = Ibw.end<uchar>();
		//float alfa = 0.95, beta = 1.1;//0.95 1.1
		for (; bw != end; ++eb, ++eg, ++er, ++bw){
			*bw = (uchar)(*er + *eg)*0.5>*eb ? (uchar)(*er + *eg)*0.5 : *eb;
			//*bw = max(max(*er, *eg), *eb);
		}
		return Ibw;
	}
	Mat tooLightP_morph(const Mat& in){
		Mat image = in.clone();
		Mat mat_thresh, mat_morph;
		Mat grad = maxGradient(in);
		threshRegion(grad, mat_thresh, 1, 6, false);//1 6
		medianBlur(mat_thresh, mat_thresh, 3);
		Mat element = getStructuringElement(MORPH_RECT, Size(23, 2));//15 2 5 5 3 ;15 2 7 7 2 5
		morphologyEx(mat_thresh, mat_morph, MORPH_CLOSE, element);
		medianBlur(mat_morph, mat_morph, 7);
		element = getStructuringElement(MORPH_RECT, Size(2, 7));//15 2 5 5 3 ;15 2 7 7 2 5
		morphologyEx(mat_morph, mat_morph, MORPH_OPEN, element);
		medianBlur(mat_morph, mat_morph, 5);
		return mat_morph;
	}
	Mat darkP_morph( const Mat& in){
		Mat gray;
		cvtColor(in, gray, CV_BGR2GRAY);
		Mat thresh;
		int now = 0, before = 0;
		vector<int> diff_low,diff_high;
		uchar i = 1;
		double h = threshold(gray, thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		
		for (int i=0; i <254; i++){
			threshold(gray, thresh, i, 255, CV_THRESH_BINARY_INV);
			now = countNonZero(thresh);
			int diff = now - before;
			if (i<h)
			   diff_low.push_back(diff);
			else 
				diff_high.push_back(diff);
			before = now;
		}
		Mat mat_diff_low;
		vector<float> temp;
		normalize(diff_low, temp, 1,0, CV_MINMAX,-1);
		Mat(temp).convertTo(mat_diff_low,CV_8UC1,255,0);
		Mat mat_diff_high;
		normalize(diff_high, temp, 1, 0, CV_MINMAX, -1);
		Mat(temp).convertTo(mat_diff_high, CV_8UC1, 255, 0);
		//cout << mat_diff_low.size() << "  " << mat_diff_high.size() << endl;
		//cout << mat_diff_low.type() << "  " << mat_diff_high.type() << endl;
		double h1 = threshold(mat_diff_low, mat_diff_low, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
		double h2 = threshold(mat_diff_high, mat_diff_high, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
		//cout << h << "  " <<h1<<"  "<<h2<<"  "<< endl;
		MatIterator_<uchar> lit = mat_diff_low.begin<uchar>();
		MatIterator_<uchar> hit = mat_diff_high.begin<uchar>();
		MatIterator_<uchar> litend =mat_diff_low.end<uchar>();
		MatIterator_<uchar> hitend = mat_diff_high.end<uchar>();
		int up=h, low=0;
		for (; ((*lit) < h1) && lit != litend; lit++){
			low++;
		}

		for (; ((*hit) < h2) && hit != hitend; hit++){
			up++;
		}
		//cout << low << "  " << up <<  endl;
		MatIterator_<uchar> it = gray.begin<uchar>();
		MatIterator_<uchar> itend = gray.end<uchar>();
		double slope = 255.0 / (up - low);
		double bias = slope*(-low);
		 //it = gray.begin<uchar>();
		for (; it != itend; ++it){
			*it = saturate_cast<uchar>(slope*(*it) + bias);
		}
		Mat mat_thresh;
		Mat mat_sober;
		Sobel(gray, mat_sober, CV_16S, 1, 0, 3);
		convertScaleAbs(mat_sober, mat_sober);
		threshold(mat_sober, mat_thresh, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		//threshRegion(mat_sober, mat_thresh, 1, 6, false);
		medianBlur(mat_thresh, mat_thresh, 7);
		Mat element;
		element = getStructuringElement(MORPH_ELLIPSE, Size(25, 5));
		morphologyEx(mat_thresh, mat_thresh, MORPH_CLOSE, element);
		return mat_thresh;

	}
	
	void findRect(const Mat &clear_gray,const Mat &binary_mat,vector<Mat> &out){
		Mat mat_morph = binary_mat;
		int cols = clear_gray.cols;
		int rows = clear_gray.rows;
		float verifyMin = max(rows*cols/ 1000.0, 1500.0);//400  1000
	    float verifyMax = (float)(rows*cols) / 8.0;//40
		vector<vector<Point>> contours;
		findContours(mat_morph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		vector<vector<Point>>::iterator itc = contours.begin();

		vector<RotatedRect> rects;
		if (contours.size() == 0)
			return;
		int low, up;
		verifyPointsNum(contours, low, up);

		while (itc != contours.end()) {
			int pointsize = (*itc).size();
			if (pointsize < low || pointsize>up){
				itc++;
				continue;
			}
			RotatedRect mr = minAreaRect(Mat(*itc));

			if (!verifySizes(mr, verifyMin, verifyMax)){
				itc++;
				continue;
			}
			rects.push_back(mr);
			++itc;
		}
		vector<vector<Point>>(contours).swap(contours);
		if (rects.empty())
			return;
		cross_rectMerge(rects, verifyMin, verifyMax);
		for (size_t m = 0; m < rects.size(); m++) {
			RotatedRect roi = rects[m];

			float r = (float)roi.size.width / (float)roi.size.height;
			float roi_angle = roi.angle;
			bool clockwise=true;
			Size roi_rect_size = roi.size;
			if (r < 1) {
				clockwise = false;
				roi_angle = -90 - roi_angle;
				swap(roi_rect_size.width, roi_rect_size.height);
				r = 1 / r;
			}
			//cout << roi_angle << endl;
			Rect_<float> safeBoundRect;
			bool isFormRect = calcSafeRect(roi, clear_gray, safeBoundRect);//blur output color candidates                    
			if (!isFormRect)
				continue;

			Mat bound_mat = clear_gray(safeBoundRect);//blur output color candidates
			Mat bound_mat_b = mat_morph(safeBoundRect);
			Point2f roi_ref_center = roi.center - safeBoundRect.tl();

			Mat rotated_mat;
			Mat rotated_mat_b;
			float angle_rotate = clockwise ? roi_angle : (-roi_angle);
			if (!rotation(bound_mat, rotated_mat, roi_rect_size, roi_ref_center,angle_rotate))
				continue;
			if (!rotation(bound_mat_b, rotated_mat_b, roi_rect_size, roi_ref_center, angle_rotate))
			   continue;
			rotated_mat = deflection(rotated_mat_b, rotated_mat);
			Mat plate_mat;
			plate_mat.create(48, 153, CV_8UC1);//36,128

			if (rotated_mat.cols >= 153 || rotated_mat.rows >= 48)
				resize(rotated_mat, plate_mat, plate_mat.size(), 0, 0, INTER_AREA);
			else
				resize(rotated_mat, plate_mat, plate_mat.size(), 0, 0);

			out.push_back(plate_mat);
		}
		return;
	}
	void  plateDetect(const Mat &in, vector<Mat> &out) {     //, vector<Rect> &vector_rects
		if (!out.empty())
			out.clear();
		if (in.type() != CV_8UC3) {
			cout << "Wrong type of input image of functioon PLATEDETECT" << endl;
			return;
		}
		Mat avg1, std1;
		Mat image = in.clone();
		meanStdDev(in, avg1, std1);
		double meanval = 100.0;
		double stdval = 80.0;
		double std = (std1.at<double>(0, 0) + std1.at<double>(1, 0) + std1.at<double>(2, 0)) / 3;
		for (int c = 0; c < 3; c++) {
			double a = stdval / std1.at<double>(c, 0);
			double b = meanval - stdval*avg1.at<double>(c, 0) / std1.at<double>(c, 0);
			//cout << a << "  " << b << endl;
			for (int y = 0; y < in.rows; y++) {
				for (int x = 0; x < in.cols; x++) {
					image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(a*in.at<Vec3b>(y, x)[c] + b);
				}
			}
		}
		int org_width = image.size().width;
		int org_height = image.size().height;
		int maxiSize = 1600;
		if (max(org_width, org_height) > maxiSize) {
			float right_R = 1.0* org_height / org_width;
			if (right_R < 1)
				resize(image, image, Size(maxiSize, static_cast<int>(maxiSize * right_R)), 0, 0);
			else
				resize(image, image, Size(static_cast<int>(maxiSize / right_R), maxiSize), 0, 0);
		}
		Mat clear_gray;
		cvtColor(image, clear_gray, CV_BGR2GRAY);
		vector<Mat> morphs;
		//morphs.push_back(sobelMorph(image));
		morphs.push_back(blue_normal_morph(image));
		//morphs.push_back(yellow_colorMorph(image));
		//morphs.push_back(tooLightP_morph(image));
		//morphs.push_back(darkP_morph(image));
#pragma omp parallel for
		for (size_t cvn = 0; cvn < morphs.size(); cvn++)
			findRect(clear_gray, morphs[cvn], out);
		return;
	}
	bool findCPlate(const cv::Mat &clear_gray, const cv::Mat &binary_mat, std::vector<CPlate> &out) {
		Mat mat_morph = binary_mat;
		int cols = clear_gray.cols;
		int rows = clear_gray.rows;
		float verifyMin = max(rows*cols / 1000.0, 1500.0);//400  1000
		float verifyMax = (float)(rows*cols) / 8.0;//40
		vector<vector<Point>> contours;
		findContours(mat_morph, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		vector<vector<Point>>::iterator itc = contours.begin();

		vector<RotatedRect> rects;
		if (contours.size() == 0)
			return false;
		int low, up;
		Locate::verifyPointsNum(contours, low, up);

		while (itc != contours.end()) {
			int pointsize = (*itc).size();
			if (pointsize < low || pointsize>up) {
				itc++;
				continue;
			}
			RotatedRect mr = minAreaRect(Mat(*itc));

			if (!Locate::verifySizes(mr, verifyMin, verifyMax)) {
				itc++;
				continue;
			}
			rects.push_back(mr);
			++itc;
		}
		vector<vector<Point>>(contours).swap(contours);
		if (rects.empty())
			return false;
		Locate::cross_rectMerge(rects, verifyMin, verifyMax);
		for (size_t m = 0; m < rects.size(); m++) {
			RotatedRect roi = rects[m];

			float r = (float)roi.size.width / (float)roi.size.height;
			float roi_angle = roi.angle;
			bool clockwise = true;
			Size roi_rect_size = roi.size;
			if (r < 1) {
				clockwise = false;
				roi_angle = -90 - roi_angle;
				swap(roi_rect_size.width, roi_rect_size.height);
				r = 1 / r;
			}
			//cout << roi_angle << endl;
			Rect_<float> safeBoundRect;
			bool isFormRect = Locate::calcSafeRect(roi, clear_gray, safeBoundRect);//blur output color candidates                    
			if (!isFormRect)
				continue;

			Mat bound_mat = clear_gray(safeBoundRect);//blur output color candidates
			Mat bound_mat_b = mat_morph(safeBoundRect);
			Point2f roi_ref_center = roi.center - safeBoundRect.tl();

			Mat rotated_mat;
			Mat rotated_mat_b;
			float angle_rotate = clockwise ? roi_angle : (-roi_angle);
			if (!Locate::rotation(bound_mat, rotated_mat, roi_rect_size, roi_ref_center, angle_rotate))
				continue;
			if (!Locate::rotation(bound_mat_b, rotated_mat_b, roi_rect_size, roi_ref_center, angle_rotate))
				continue;
			rotated_mat = Locate::deflection(rotated_mat_b, rotated_mat);
			Mat plate_mat;
			plate_mat.create(48, 153, CV_8UC1);//36,128 102 32

			if (rotated_mat.cols >= 153 || rotated_mat.rows >= 48)
				resize(rotated_mat, plate_mat, plate_mat.size(), 0, 0, INTER_AREA);
			else
				resize(rotated_mat, plate_mat, plate_mat.size(), 0, 0);
			CPlate plate;
			plate.plateMat = plate_mat;
			plate.plateRotatedRect = roi;
			out.push_back(plate);
		}
		if (out.empty())
			return false;
		return true;
	}
	void  CplateDetect(const Mat &in, vector<CPlate> &out,const std::vector<bool> &methods) {     //, vector<Rect> &vector_rects
		if (!out.empty())
			out.clear();
		if (in.type() != CV_8UC3) {
			cout << "Wrong type of input image of functioon PLATEDETECT" << endl;
			return;
		}
		Mat avg1, std1;
		Mat image = in.clone();
		meanStdDev(in, avg1, std1);
		double meanval = 100.0;
		double stdval = 80.0;
		double std = (std1.at<double>(0, 0) + std1.at<double>(1, 0) + std1.at<double>(2, 0)) / 3;
		for (int c = 0; c < 3; c++) {
			double a = stdval / std1.at<double>(c, 0);
			double b = meanval - stdval*avg1.at<double>(c, 0) / std1.at<double>(c, 0);
			//cout << a << "  " << b << endl;
			for (int y = 0; y < in.rows; y++) {
				for (int x = 0; x < in.cols; x++) {
					image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(a*in.at<Vec3b>(y, x)[c] + b);
				}
			}
		}
		int org_width = image.size().width;
		int org_height = image.size().height;
		int maxiSize = 1600;
		if (max(org_width, org_height) > maxiSize) {
			float right_R = 1.0* org_height / org_width;
			if (right_R < 1)
				resize(image, image, Size(maxiSize, static_cast<int>(maxiSize * right_R)), 0, 0);
			else
				resize(image, image, Size(static_cast<int>(maxiSize / right_R), maxiSize), 0, 0);
		}
		Mat clear_gray;
		cvtColor(image, clear_gray, CV_BGR2GRAY);
		vector<Mat> morphs;
		//morphs.push_back(sobelMorph(image));
		morphs.push_back(blue_normal_morph(image));
		//morphs.push_back(yellow_colorMorph(image));
		//morphs.push_back(tooLightP_morph(image));
		//morphs.push_back(darkP_morph(image));
#pragma omp parallel for
		for (size_t cvn = 0; cvn < morphs.size(); cvn++) {
			imwrite("debug_morph.jpg", morphs[0]);
			findCPlate(clear_gray, morphs[cvn], out);
		}
		return;
	}
}

#endif