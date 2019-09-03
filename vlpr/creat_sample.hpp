#ifndef CREAT_SAMPLE
#define CREAT_SAMPLE
#include "platelocate.hpp"
#include "mser2.hpp"

std::vector<cv::Mat> mser3Rect(const cv::Mat& src){
	cv::Mat image = src.clone();

	std::vector<std::vector<cv::Point>> all_contours;
	std::vector<cv::Rect> all_boxes;
	all_contours.reserve(256);
	all_boxes.reserve(256);

	cv::Ptr<cv::MSER2> mser;
	mser = cv::MSER2::create();

	mser->detectBrightRegions(image, all_contours, all_boxes);

	std::vector<cv::Mat> char_candi;
	std::cout << all_contours.size() << std::endl;
#pragma omp parallel for
	for (int i = 0; i < all_contours.size(); i++){
		cv::RotatedRect rrect = cv::minAreaRect(all_contours[i]);
		float angle = rrect.angle;
		int area = rrect.size.height*rrect.size.width;
		float r = ((float)rrect.size.height) / rrect.size.width;
		cv::Size roi_size = rrect.size;
		if (r < 1){
			angle = -(angle + 90);
			swap(roi_size.height, roi_size.width);
			r = 1 / r;
		}

		if ((angle + 45 < 0) ||(r >5) || (area >1500))
			continue;
		else{
			Rect_<float> safeBoundRect;
			bool isFormRect = Locate::calcSafeRect(rrect, image, safeBoundRect);
			cv::Point center = rrect.center - safeBoundRect.tl();
			Mat rotated_char;
			if (!Locate::rotation(image(safeBoundRect), rotated_char, roi_size, center, angle))
				continue;
			cv::threshold(rotated_char, rotated_char, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
			//cv::Size char_size = cv::Size(44, 44);
			float ratio = rotated_char.cols / rotated_char.rows;
			cv::resize(rotated_char, rotated_char, cv::Size(44, int(44 * ratio)), 0, 0);
			int expendW = (44 - rotated_char.cols) / 2;
			cv::Mat out(44, 44, CV_8UC1, cv::Scalar(0));
			Mat outRoi = out(Rect(expendW, 0, rotated_char.cols, rotated_char.rows));
			addWeighted(outRoi, 0, rotated_char, 1, 0, out);
			char_candi.push_back(out);
		}

		
	}
	return char_candi;
}


#endif