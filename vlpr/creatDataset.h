#ifndef CREATDATASET
#define CREATDATASET

#include "util.hpp"
#include <string>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "mser2.hpp"
#include "platelocate.hpp"



//using namespace cv;

cv::Mat mser1Operate(const cv::Mat &src){
	cv::Mat image = src.clone();
	std::vector<std::vector<cv::Point>> mserContours;
	std::vector<cv::Rect> boxes;

	cv::Ptr<cv::MSER> mser1 = cv::MSER::create(1, 100, 1000, 0.35);//1 100 1000 0.35
	mser1->detectRegions(image, mserContours, boxes);
#pragma omp parallel for
	for (int i = 0; i < mserContours.size(); i++)
		ellipse(image, fitEllipse(mserContours[i]), cv::Scalar(0));

	return image;
}

cv::Mat mser2Test(const cv::Mat& src){
	cv::Mat image = src.clone();

	std::vector<std::vector<std::vector<cv::Point>>> all_contours;
	std::vector<std::vector<cv::Rect>> all_boxes;
	all_contours.at(0).reserve(256);
	all_contours.at(1).reserve(256);
	all_boxes.resize(2);
	all_boxes.at(0).reserve(256);
	all_boxes.at(1).reserve(256);

	cv::Ptr<cv::MSER2> mser;
	mser = cv::MSER2::create();

	mser->detectRegions(image, all_contours.at(0), all_boxes.at(0), all_contours.at(1), all_boxes.at(1));


#pragma omp parallel for
	for (int color_index = 0; color_index < 2; color_index++){
		//#pragma omp parallel for
		for (int i = 0; i < all_contours.at(color_index).size(); i++)
			ellipse(image, fitEllipse(all_contours.at(color_index)[i]), cv::Scalar(0));

	}
	return image;
}

cv::Mat mser2DarkTest(const cv::Mat& src){
	cv::Mat image = src.clone();

	std::vector<std::vector<cv::Point>> all_contours;
	std::vector<cv::Rect> all_boxes;
	all_contours.reserve(256);
	all_boxes.reserve(256);

	cv::Ptr<cv::MSER2> mser;
	mser = cv::MSER2::create();

	mser->detectDarkRegions(image, all_contours, all_boxes);

#pragma omp parallel for
	for (int i = 0; i < all_contours.size(); i++)
		ellipse(image, fitEllipse(all_contours[i]), cv::Scalar(255));
	return image;
}

cv::Mat mser2BrightTest(const cv::Mat& src){
	cv::Mat image = src.clone();

	std::vector<std::vector<cv::Point>> all_contours;
	std::vector<cv::Rect> all_boxes;
	all_contours.reserve(256);
	all_boxes.reserve(256);

	cv::Ptr<cv::MSER2> mser;
	mser = cv::MSER2::create();

	mser->detectBrightRegions(image, all_contours, all_boxes);

#pragma omp parallel for
	for (int i = 0; i < all_contours.size(); i++)
		ellipse(image, fitEllipse(all_contours[i]), cv::Scalar(255));
	return image;
}
void calcuResize(){
	//std::vector<std::string> folder = { "0","2","3","4","5","6","7","8","9","A", "B", //"C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z" };
	/*std::vector<std::string> folder = { "1 (2)", "1 (3)", "1 (4)", "1 (5)", "1 (6)", "1 (7)",
		"1 (8)", "1 (9)", "1 (10)", "1 (11)", "1 (12)", "1 (13)", "1 (14)", "1 (15)", "1 (16)",
		"1 (17)", "1 (18)", "1 (19)", "1 (20)", "1 (21)","1 (22)", "1 (23)", "1 (24)", "1 (25)", 
		"1 (26)", "1 (27)", "1 (28)", "1 (29)", "1 (30)", "1 (31)", "1 (32)" };*/
	std::vector<std::string> folder = { "has1","has2","has3" };
	std::string a ="../resources/train/charsJudge/";
	std::string dst_path = ("../resources/train/charsJudge/has/");
	std::vector<std::string> train_list;
	for (auto i = 0; i < folder.size(); i++){
		train_list=utils::getFiles(a + folder[i]);	
		cv::Mat image;
		std::string imageName;
		for (size_t j = 0; j < train_list.size(); j++){
			if (j %2== 0){
				imageName = train_list[j].c_str();
				image = cv::imread(imageName);
				//cv::resize(image, image, Size(24, 24), 0, 0);
				imageName = Utils::getFileName(imageName, 1);
				imwrite(dst_path + imageName, image);
			}
		}
	}
}
void calcuResize2(){
	std::vector<std::string> folder = { "no1", "no2"};
	std::string a = "../resources/train/charsJudge/";
	std::string dst_path = ("../resources/train/charsJudge/no/");
	std::vector<std::string> train_list;
		train_list = utils::getFiles(a + folder[0]);
		cv::Mat image;
		std::string imageName;
		for (size_t j = 0; j < train_list.size(); j++){
			if (j % 8 == 0 || j % 8 == 3 || j % 5 == 0){
				imageName = train_list[j].c_str();
				image = cv::imread(imageName);
				//cv::resize(image, image, Size(24, 24), 0, 0);
				imageName = Utils::getFileName(imageName, 1);
				imwrite(dst_path + imageName, image);
			}
		}
		train_list = utils::getFiles(a + folder[1]);
		for (size_t j = 0; j < train_list.size(); j++){
			if (j % 2== 0 ){
				imageName = train_list[j].c_str();
				image = cv::imread(imageName);
				//cv::resize(image, image, Size(24, 24), 0, 0);
				imageName = Utils::getFileName(imageName, 1);
				imwrite(dst_path + imageName, image);
			}
		}
}
void calculate(const std::string &datasetName){
	string src_path = "../resources/image/";
	auto files = Utils::getFiles(src_path + "lifeCar/1");
	std::string dst_path = (src_path + datasetName + "/");
	size_t size = files.size();
	if (0 == size){
		std::cout << "No File Found��" << std::endl;
		return;
	}
	else{
		std::cout << "Begin to process sobelDet" << std::endl;
		cv::Mat img;
		std::string imageName;
		for (size_t i = 0; i < size; i++){
			imageName = files[i].c_str();
			img = cv::imread(imageName);
			if (img.type() != CV_8UC3)
				continue;
			Mat avg1, std1;
			Mat image = img.clone();
			meanStdDev(img, avg1, std1);
			double meanval = 100.0;
			double stdval = 80.0;
			double std = (std1.at<double>(0, 0) + std1.at<double>(1, 0) + std1.at<double>(2, 0)) / 3;
			for (int c = 0; c < 3; c++){
				double a = stdval / std1.at<double>(c, 0);
				double b = meanval - stdval*avg1.at<double>(c, 0) / std1.at<double>(c, 0);
				//cout << a << "  " << b << endl;
				for (int y = 0; y < img.rows; y++){
					for (int x = 0; x < img.cols; x++){
						image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(a*img.at<Vec3b>(y, x)[c] + b);
					}
				}
			}
			int org_width = image.size().width;
			int org_height = image.size().height;
			int maxiSize = 1600;
			if (max(org_width, org_height) > maxiSize){
				float right_R = 1.0* org_height / org_width;
				if (right_R < 1)
					resize(image, image, Size(maxiSize, static_cast<int>(maxiSize * right_R)), 0, 0);
				else
					resize(image, image, Size(static_cast<int>(maxiSize / right_R), maxiSize), 0, 0);
			}
			Mat miniImg;
			cv::resize(image, miniImg, Size(80, 80), 0, 0);
			imageName = Utils::getFileName(imageName, 1);
			imwrite(dst_path + imageName, miniImg);
			/*cv::cvtColor(miniImg, miniImg, CV_RGB2HSV);
			vector<int> feature;
			for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
			Mat roi;
			miniImg(Rect(i * 20, j * 20, 20, 20)).copyTo(roi);
			Mat avg1, std1;
			meanStdDev(roi, avg1, std1);
			feature.push_back(static_cast<int>(avg1.at<double>(1, 0)));
			feature.push_back(static_cast<int>(std1.at<double>(1, 0)));
			feature.push_back(static_cast<int>(avg1.at<double>(2, 0)));
			feature.push_back(static_cast<int>(std1.at<double>(2, 0)));
			}
			}
			std::cout <<Mat(feature).t() << std::endl;*/

		}
		/*Mat avg1, std1;
		meanStdDev(img, avg1, std1);
		allSatuMean.push_back(static_cast<int>(avg1.at<double>(1,0)));
		allSatuStdD.push_back(static_cast<int>(std1.at<double>(1,0)));
		allValMean.push_back(static_cast<int>(avg1.at<double>(2,0)));
		allValStdD.push_back(static_cast<int>(std1.at<double>(2, 0)));
		}
		cv::Mat SatuMeanAvg, SatuMeanStdD;
		//std::vector<vector<int>> allCompu = { allSatuMean, allSatuStdD, allValMean, allValStdD };
		meanStdDev(allSatuMean, SatuMeanAvg, SatuMeanStdD);
		std::cout << SatuMeanAvg << "      " <<SatuMeanStdD << std::endl;
		meanStdDev(allSatuStdD, SatuMeanAvg, SatuMeanStdD);
		std::cout << SatuMeanAvg << "      " << SatuMeanStdD << std::endl;
		meanStdDev(allValMean, SatuMeanAvg, SatuMeanStdD);
		std::cout << SatuMeanAvg << "      " << SatuMeanStdD << std::endl;
		meanStdDev(allValStdD, SatuMeanAvg, SatuMeanStdD);
		std::cout << SatuMeanAvg << "      " << SatuMeanStdD << std::endl;*/
		return;
	}

}
void creatMserSet(const std::string &datasetName){
	string src_path = "../resources/image/";
	auto files = Utils::getFiles(src_path + "lifeCar/my-iphone");
	std::string dst_path = (src_path + datasetName + "/");
	size_t size = files.size();
	if (0 == size){
		std::cout << "No File Found��" << std::endl;
		return;
	}
	else{
		std::cout << "Begin to process sobelDet" << std::endl;
		cv::Mat img;
		std::string imageName;
		for (size_t i = 0; i < size; i++){
			imageName = files[i].c_str();
			std::cout << "--------" << imageName << std::endl;
			img = cv::imread(imageName);
			vector<Mat> candidates;
			Locate::plateDetect(img, candidates);
			if (candidates.empty()){
				std::cout << "No plate candidates!!!" << std::endl;
				continue;
			}
			imageName = Utils::getFileName(imageName, 0);
			char file[100];
			imwrite(dst_path + imageName + ".jpg", candidates[0]);
			for (int j = 1; j < candidates.size(); j++){
				sprintf_s(file, "candi%d.jpg", j);
				imwrite(dst_path + imageName + file, candidates[j]);
			}
		}
		return;
	}
}

void mser1Test(const std::string &datasetName){
	string src_path = "../resources/image/";
	auto files = Utils::getFiles(src_path + "sense/a1");//lifeCar/combineP
	std::string dst_path = (src_path + datasetName + "/");
	std::cout << "Begin to process msertest" << std::endl;
	cv::Mat img, mser2;
	std::string imageName;
	for (size_t i = 0; i < files.size(); i++){
		imageName = files[i].c_str();
		std::cout << "--------" << imageName << std::endl;
		img = cv::imread(imageName, 0);
		imageName = Utils::getFileName(imageName, 1);
		cv::Canny(img, mser2, 100, 200, 3);
		//mser2 = mser2DarkTest(img);
		imwrite(dst_path + imageName, mser2);
	}
	return;
}

/*void creatDataSet(const std::string &datasetName){

auto files = Utils::getFiles(src_path + "msertest");
std::string dst_path = (src_path + datasetName + "/");
size_t size = files.size();
if (0 == size){
std::cout << "No File Found!!" << std::endl;
return;
}
else{
std::cout << "Begin to process sobelDet" << std::endl;
cv::Mat img, mser2;
std::string imageName;
for (size_t i = 0; i < size; i++){
imageName = files[i].c_str();
std::cout << "--------" << imageName << std::endl;
img = cv::imread(imageName, 0);
imageName = Utils::getFileName(imageName, 1);
mser2 = mser2BrightTest(img);
imwrite(dst_path + imageName, mser2);

}
return;
}
}*/

void  imageProcess1(const Mat& src, std::vector<cv::RotatedRect> &rrects){
	Mat image = src.clone();
	std::vector<std::vector<Point>>all_contours;
	std::vector<cv::Rect> all_boxes;
	all_contours.reserve(120);
	all_boxes.reserve(120);
	cv::Ptr<cv::MSER2> mser;
	mser = cv::MSER2::create();

	mser->detectBrightRegions(image, all_contours, all_boxes);

#pragma omp parallel for
	for (int i = 0; i < all_contours.size(); i++){
		cv::RotatedRect rrect = cv::minAreaRect(all_contours[i]);
		rrects.push_back(rrect);
	}
	//std::cout << rrects.size();
	return;
}

void makeChar(const Mat& image, std::vector<cv::Mat> &char_rect){
	std::vector<cv::RotatedRect> rrects;
	imageProcess1(image, rrects);
#pragma omp parallel for
	for (int i = 0; i < rrects.size(); i++){
		cv::RotatedRect rrect = rrects[i];
		float angle = rrect.angle;
		int area = rrect.size.height*rrect.size.width;
		float r = ((float)rrect.size.height) / rrect.size.width;
		cv::Size roi_size=rrect.size;
		bool clockwise = true;
		if (r < 1){
			angle = -(angle + 90);
			swap(roi_size.height, roi_size.width);
			r = 1 / r;
			clockwise = false;
		}
		if ((angle + 45 < 0) || (r >7) || (area >1000)||area<80)
			continue;
		else{
			roi_size.width *= (int)1.4;
			roi_size.height *= (int)1.4;
			cv::Rect_<float> safeBoundRect;
			bool isFormRect = Locate::calcSafeBigRect(rrect, image, safeBoundRect);
			cv::Point center = rrect.center - safeBoundRect.tl();
			cv::Mat rotated_char;
			float rotate_angle = (clockwise) ? angle : (-angle);
			if (!Locate::rotation(image(safeBoundRect), rotated_char, roi_size, center, rotate_angle)) {
				//std::cout << "Wrong rotation" << std::endl;
				continue;
			   }
			cv::threshold(rotated_char, rotated_char, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
			//cv::Size char_size = cv::Size(44, 44);
			float ratio = (float)rotated_char.cols / (float)rotated_char.rows;
			cv::resize(rotated_char, rotated_char, cv::Size(int(24 * ratio), 24), 0, CV_INTER_AREA);
			int expendW = (24 - rotated_char.cols) / 2;
			cv::Mat out(24, 24, CV_8UC1, cv::Scalar(0));
			cv::Mat outRoi = out(cv::Rect(expendW, 0, rotated_char.cols, rotated_char.rows));
			rotated_char.copyTo(outRoi);
			char_rect.push_back(out);
		}
	}
	return;
}


void creatCharSet(){
	string src_path = "../resources/image/sense/";
	vector<vector<string>> fileAll;
	auto files1 = Utils::getFiles(src_path + "iphoneCarplate");
	auto files2 = Utils::getFiles(src_path + "iphoneNoplate");
	auto files3 = Utils::getFiles(src_path + "badPlate");
	fileAll.push_back(files1);
	fileAll.push_back(files2);
	fileAll.push_back(files3);
	for (size_t ii = 0; ii < 3; ii++){
		//auto files = Utils::getFiles(src_path + "testSet");
		auto files = fileAll[ii];
		char file2[100];
		sprintf_s(file2, "a%d", (int)ii);
		std::string dst_path = (src_path + file2 + "/");
		size_t size = files.size();
		if (0 == size){
			std::cout << "No File Found!!" << std::endl;
			continue;
		}
		else{
			std::cout << "Begin to process charCandiSet" << std::endl;
			for (size_t i = 0; i < size; i++){
				std::string imageName = files[i].c_str();
				//std::cout << "--------" << imageName << std::endl;

				cv::Mat img = cv::imread(imageName, 0);
				std::vector<cv::Mat> candidates;
				candidates.reserve(120);
				makeChar(img, candidates);
				if (candidates.empty())
					continue;
				imageName = Utils::getFileName(imageName, 0);
				char file[100];
				for (int j = 0; j < candidates.size(); j++){
					sprintf_s(file, "%d.jpg", j);
					cv::imwrite(dst_path + imageName + file, candidates[j]);
				}
			}
		}
	}
}

#endif