//////////////////////////////////////////////////////////////////////////
// Desciption:
// Abstract classes for car plate.(CCharacter,CPlate)
//////////////////////////////////////////////////////////////////////////
#ifndef VLPR_CORE_BASIC_CLASS_H_
#define VLPR_CORE_BASIC_CLASS_H_

#include "opencv2/opencv.hpp"
#include "config.h"

class CPlate {
public:
	CPlate() {
		isPlate = false;
		plateRotatedRect = cv::RotatedRect(cv::Point(0, 0), cv::Size(0, 0), 0.0);
		plateMat = cv::Mat();
	}

	CPlate& operator=(const CPlate& other) {
		if (this != &other) {
			plateMat = other.plateMat;
			isPlate = other.isPlate;
			plateStr = other.plateStr;
			plateRotatedRect = other.plateRotatedRect;
			charMats = other.charMats;
			is1 = other.is1;
		}
		return *this;
	}

	cv::Mat plateMat;
	std::string plateStr;
	bool isPlate;
	cv::RotatedRect plateRotatedRect;
	std::vector<cv::Mat> charMats;
	std::vector <bool> is1;
};

#endif  // EASYPR_CORE_PLATE_H_