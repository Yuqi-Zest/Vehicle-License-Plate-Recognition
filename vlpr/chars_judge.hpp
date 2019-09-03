#ifndef CHARS_JUDGE_HPP
#define CHARS_JUDGE_HPP
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include<opencv2/opencv.hpp>
#include "creatDataset.h"
#include <opencv2/ml/ml.hpp>
#include<algorithm>
#include "erfilter.hpp"
/*typedef enum {
	kForward = 1, // correspond to "has plate"
	kInverse = 0  // correspond to "no plate"
} SvmLabel;*/
namespace charsJudge{
	void gridCSearch(cv::Mat &samples, std::vector<int> &labels, cv::ml::ParamGrid &Cgrid, cv::ml::ParamGrid &Ggrid, std::string& xmlFile,bool visualize = true){

		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::RBF);
		//float a[2] = { 1, 3 };
		//cv::Mat s(1, 2, CV_32F, a);
		//s.reshape(2, 1);
		//std::cout << s << std::endl;
		//svm->setClassWeights();
		// 1.4 bug fix: old 1.4 ver gamma is 1
		svm->setGamma(0.09);//0.1
		svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));

		double Cmin = Cgrid.minVal;
		double Cmax = Cgrid.maxVal;
		double Cstep = Cgrid.logStep;
		double Gmin = Ggrid.minVal;
		double Gmax = Ggrid.maxVal;
		double Gstep = Ggrid.logStep;

		std::FILE *fp;
		fopen_s(&fp,"../resources/train/charsJudge/gridsvm.txt","w+");
		if (fp == NULL){
			std::cout << "can not open file" << std::endl;
			return;
		}
		std::vector<double> CvalV;
		std::vector<double> GvalV;
		std::vector<double> accuracyV;
		std::vector<double> preciseV;
		std::vector<double> recallV;
		int timeC = 0;
		int timeG = 0;
		double bestC=0, bestG=0, bestAccuracy=0;
		for (double Cval = Cmin; Cval <= Cmax; Cval *= Cstep){
			++timeC;
			timeG = 0;
			for (double Gval = Gmin; Gval <= Gmax; Gval *= Gstep){
				++timeG;
				std::fprintf(fp, "C value is:%6.6lf\n", Cval);
				std::fprintf(fp, "G value is:%6.6lf\n", Gval);
				int crossNum = 5;
				int a = samples.rows / crossNum;
				//std::cout << "a nunber:" << a << std::endl;
				double accuracyS = 0, recallS = 0, preciseS = 0;
				for (int i = 0; i < crossNum; i++){
					cv::Mat tempTrain;
					tempTrain.push_back(samples.rowRange((i + 1)*a, samples.rows));
					std::vector<int> temp_trainL;
					for (size_t m = (a*i + a); m != labels.size(); m++)
						temp_trainL.push_back(labels[m]);

					if (i != 0){
						tempTrain.push_back(samples.rowRange(0, i*a));
						for (size_t n = 0; n < i*a; n++)
							temp_trainL.push_back(labels[n]);
					}
					svm->setC(Cval);
					svm->setGamma(Gval);
					auto train_data = cv::ml::TrainData::create(tempTrain, cv::ml::SampleTypes::ROW_SAMPLE, temp_trainL);
					svm->train(train_data);
					int ptrue_rtrue = 0;
					int ptrue_rfalse = 0;
					int pfalse_rtrue = 0;
					int pfalse_rfalse = 0;
					for (int j = 0; j < a; j++){
						int real = labels[a*i + j];
						int predict = svm->predict(samples.rowRange(a*i + j, a*i + j + 1));
						if (predict == kForward && real == kForward) ptrue_rtrue++;
						else if (predict == kForward && real == kInverse) ptrue_rfalse++;
						else if (predict == kInverse && real == kForward) pfalse_rtrue++;
						else if (predict == kInverse && real == kInverse) pfalse_rfalse++;
					}
					double accuracy = 1.0*(ptrue_rtrue + pfalse_rfalse) / a;
					accuracyS += (accuracy / crossNum);
					double recall;
					if (ptrue_rtrue + pfalse_rtrue != 0)
						recall = 1.0*ptrue_rtrue / (ptrue_rtrue + pfalse_rtrue);
					else recall = 0;
					recallS += (recall / crossNum);
					double precise;
					if (ptrue_rtrue + ptrue_rfalse != 0)
						precise = 1.0*ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse);
					else precise = 0;
					preciseS += (precise / crossNum);
					fprintf(fp, "%6.6lf , %6.6lf , %6.6lf \t\n", accuracy, recall, precise);
					/*std::cout << ptrue_rtrue << std::endl;
					std::cout << pfalse_rtrue << std::endl;
					std::cout << ptrue_rfalse << std::endl;
					std::cout << pfalse_rfalse << std::endl;*/
				}

				if (visualize){
					std::cout << "Cval   " << Cval << std::endl;
					std::cout << "Gval   " << Gval << std::endl;
					std::cout << "accuracy   " << accuracyS << std::endl;
					std::cout << "recall   " << recallS << std::endl;
					std::cout << "precise   " << preciseS << std::endl;
				}
				if (accuracyS>bestAccuracy){
					bestC = Cval;
					bestG = Gval;
				}
				CvalV.push_back(Cval);
				GvalV.push_back(Gval);
				accuracyV.push_back(accuracyS);
				preciseV.push_back(preciseS);
				recallV.push_back(recallS);
				fprintf(fp, " error= %6.4lf, recall= %6.4lf, precise= %6.4lf\n", accuracyS, recallS, preciseS);
			}
		}
		svm->setC(bestC);
		svm->setGamma(bestG);
		auto train_data1 = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE, labels);
		svm->train(train_data1);
		svm->save(xmlFile);
		cv::Mat OrigMat(timeC, timeG, CV_64FC4);//(timeG, timeC, CV_64FC4);
		cv::Mat ColorMat=cv::Mat::zeros(timeC * 8, timeG* 8, CV_8UC3);
		for (auto x = 0; x < OrigMat.rows; x++){
			for (auto y = 0; y < OrigMat.cols; y++){
				size_t mm = x*OrigMat.cols + y;
				OrigMat.at<cv::Vec4d>(x, y)[0] = CvalV[mm];
				OrigMat.at<cv::Vec4d>(x, y)[1] = GvalV[mm];
				OrigMat.at<cv::Vec4d>(x, y)[2] = preciseV[mm];
				OrigMat.at<cv::Vec4d>(x, y)[3] = recallV[mm];
				for (auto z = 0; z <8; z++){
					for (auto k = 0; k < 8; k++){
						int xn = x * 8 + z;
						int yn = y * 8 + k;
						ColorMat.at<cv::Vec3b>(xn, yn)[0] = static_cast<uchar>(preciseV[mm] * 255);
						ColorMat.at<cv::Vec3b>(xn, yn)[1] = static_cast<uchar>(accuracyV[mm] * 255);
						ColorMat.at<cv::Vec3b>(xn, yn)[2] = static_cast<uchar>(recallV[mm] * 255);
					}
				}
			}
		}
		cv::FileStorage fsGrid("../resources/train/charsJudge/OrigMat.xml", cv::FileStorage::WRITE);
		fsGrid << "OrigMat" << OrigMat;
		//cv::normalize(ColorMat, ColorMat, CV_MINMAX);
		//ColorMat.convertTo(ColorMat, 0, 255, 0);
		cv::imwrite("../resources/train/charsJudge/Color.jpg", ColorMat);
		fsGrid.release();
		return;
	}
	void creatLibsvmData(std::string libFile,bool notForTrain = 0)
		//cv::Mat &samples,std::vector<int> &responses,)
	{
		std::vector<std::pair<std::string, int>> train_file_list;
		std::vector<std::string> folder = { "color", "sober", "dark" };
		std::string a = "../resources/image/sense/";//"../resources/train/charsJudge/";
		std::vector<std::string> train_list;
		cv::Mat samples;
		std::vector<int> responses;
		for (auto i = 0; i < folder.size(); i++){
			train_list = utils::getFiles(a + folder[i]);
			for (auto file : train_list)
				train_file_list.push_back(make_pair(file, (int)i));
		}
		std::random_shuffle(train_file_list.begin(), train_file_list.end());
		std::vector<float> vec1, vec2;
		for (auto f : train_file_list) {
			auto image = cv::imread(f.first);
			if (!image.data) {
				fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
				continue;
			}
			cv::Mat temp1;
			cv::cvtColor(image, temp1, CV_RGB2GRAY);
			vector<float> feature;
			cv::Mat avg1, std1;
			cv::meanStdDev(temp1, avg1, std1);
			vec1.push_back(avg1.at<double>(0, 0));
			feature.push_back(avg1.at<double>(0, 0));
			cv::cvtColor(image, temp1, CV_RGB2HSV);
			cv::meanStdDev(temp1, avg1, std1);
			vec2.push_back(avg1.at<double>(2, 0));
			feature.push_back(avg1.at<double>(2, 0));
			responses.push_back(f.second);
			
		}
		cv::Mat temp;
		cv::normalize(cv::Mat(vec1), temp,1, 0, CV_MINMAX);
		if (notForTrain)
			temp.convertTo(temp, CV_8UC1, -140, 140);
		else
			temp.convertTo(temp, CV_32FC1, -140,140 );//均值
		samples.push_back(temp.t());

		cv::normalize(cv::Mat(vec2), temp, 1, 0, CV_MINMAX);
		if (notForTrain)
			temp.convertTo(temp, CV_8UC1, -140,140);
		else
			temp.convertTo(temp, CV_32FC1, -140, 140);//方差
		samples.push_back(temp.t());

		samples=samples.t();
		//std::cout << samples << std::endl;
		if (notForTrain){
			std::FILE *fp;
			fopen_s(&fp, (a + libFile).c_str(), "w+");
			cv::Mat canves = cv::Mat::zeros(1000, 1000, CV_8UC3);
			cv::Vec3b green(0, 255, 0), blue(255, 0, 0), red(0, 0, 255), black(0, 0, 0);
			for (int i = 0; i < samples.rows; i++){
				int aa = responses[i];
				uchar* data = samples.ptr<uchar>(i);
				cv::Rect rect(data[0] * 10, data[1] * 10, 10, 10);
		
				switch (aa){
				case 0:
					cv::rectangle(canves, rect, green, -1);
					break;
				case 1:
					cv::rectangle(canves, rect, blue, -1);
					break;
				case 2:
					cv::rectangle(canves, rect, red, -1);
					break;
				default:
					break;
				}
				fprintf(fp, "%d ", a);
				for (auto itF = 0; itF != samples.cols; itF++){
					fprintf(fp, "%d:%d  ", itF, data[itF]);
				}
				fputs("\n", fp);
			}
			cv::copyMakeBorder(canves, canves, 15, 15, 15, 15, cv::BORDER_CONSTANT, cv::Scalar(0));
			cv::imwrite(a + "canve.jpg", canves);
			fclose(fp);
		}
 		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::SIGMOID);
		svm->setDegree(10.0);//0.1
		svm->setGamma(0.014);//0.1
		svm->setCoef0(1.0);//0.1
		svm->setC(14.0);//1
		svm->setNu(0.5);//0.1
		svm->setP(1.0);//0.1
		svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));
		auto train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
			responses);
		fprintf(stdout, ">> Training SVM model, please wait...\n");
		cv::ml::ParamGrid Cgrid(0.01, 1000, 5);
		cv::ml::ParamGrid Ggrid(1e-5, 0.9, 5);
		cv::ml::ParamGrid grid(0, 0, 0);
		svm->trainAuto(train_data, 10, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),
			cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), grid, grid, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF), grid, false);
		svm->save((a+"Ssvm.xml").c_str());
		//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load<cv::ml::SVM>(a+"Csvm.xml");
		cv::Mat pred1=cv::imread((a + "canve.jpg").c_str());
		cv::Mat pred = pred1(cv::Rect(15,15,1000,1000));
		for (int i = 0; i < pred.rows; i++){
			cv::Vec3b hg(193,255,193),hb(222,196,176), hr(122,160,255), black(0, 0, 0);
			for (int j = 0; j < pred.cols; j++){
				float predArray[2] = { i/10, j/10 };
				cv::Mat predmat = cv::Mat(1, 2, CV_32FC1, predArray);
				int aa = svm->predict(predmat);
				cv::Rect rect(i,j,10,10);
				switch (aa){
				case 0:
					cv::rectangle(pred,rect,hg,-1);
					break;
				case 1:
					cv::rectangle(pred, rect, hb, -1);
					break;
				case 2:
					cv::rectangle(pred, rect, hr, -1);
					break;
				default:
					break;
				}
			}
		}
		cv::copyMakeBorder(pred, pred, 15, 15, 15, 15, cv::BORDER_CONSTANT, cv::Scalar(0));
		
		cv::imwrite(a+"Spred.jpg",pred);
		return;
	}
	
	bool guo_hall_thinning(const cv::Mat1b & img, cv::Mat& skeleton)
	{

		uchar* img_ptr = img.data;
		uchar* skel_ptr = skeleton.data;

		for (int row = 0; row < img.rows; ++row)
		{
			for (int col = 0; col < img.cols; ++col)
			{
				if (*img_ptr)
				{
					int key = row * img.cols + col;
					if ((col > 0 && *img_ptr != img.data[key - 1]) ||
						(col < img.cols - 1 && *img_ptr != img.data[key + 1]) ||
						(row > 0 && *img_ptr != img.data[key - img.cols]) ||
						(row < img.rows - 1 && *img_ptr != img.data[key + img.cols]))
					{
						*skel_ptr = 255;
					}
					else
					{
						*skel_ptr = 128;
					}
				}
				img_ptr++;
				skel_ptr++;
			}
		}

		int max_iters = 10000;
		int niters = 0;
		bool changed = false;

		/// list of keys to set to 0 at the end of the iteration
		std::deque<int> cols_to_set;
		std::deque<int> rows_to_set;

		while (changed && niters < max_iters)
		{
			changed = false;
			for (unsigned short iter = 0; iter < 2; ++iter)
			{
				uchar *skeleton_ptr = skeleton.data;
				rows_to_set.clear();
				cols_to_set.clear();
				// for each point in skeleton, check if it needs to be changed
				for (int row = 0; row < skeleton.rows; ++row)
				{
					for (int col = 0; col < skeleton.cols; ++col)
					{
						if (*skeleton_ptr++ == 255)
						{
							bool p2, p3, p4, p5, p6, p7, p8, p9;
							p2 = (skeleton.data[(row - 1) * skeleton.cols + col]) > 0;
							p3 = (skeleton.data[(row - 1) * skeleton.cols + col + 1]) > 0;
							p4 = (skeleton.data[row     * skeleton.cols + col + 1]) > 0;
							p5 = (skeleton.data[(row + 1) * skeleton.cols + col + 1]) > 0;
							p6 = (skeleton.data[(row + 1) * skeleton.cols + col]) > 0;
							p7 = (skeleton.data[(row + 1) * skeleton.cols + col - 1]) > 0;
							p8 = (skeleton.data[row     * skeleton.cols + col - 1]) > 0;
							p9 = (skeleton.data[(row - 1) * skeleton.cols + col - 1]) > 0;

							int C = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
								(!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
							int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
							int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
							int N = N1 < N2 ? N1 : N2;
							int m = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

							if ((C == 1) && (N >= 2) && (N <= 3) && (m == 0))
							{
								cols_to_set.push_back(col);
								rows_to_set.push_back(row);
							}
						}
					}
				}

				// set all points in rows_to_set (of skel)
				unsigned int rows_to_set_size = (unsigned int)rows_to_set.size();
				for (unsigned int pt_idx = 0; pt_idx < rows_to_set_size; ++pt_idx)
				{
					if (!changed)
						changed = (skeleton.data[rows_to_set[pt_idx] * skeleton.cols + cols_to_set[pt_idx]]) > 0;

					int key = rows_to_set[pt_idx] * skeleton.cols + cols_to_set[pt_idx];
					skeleton.data[key] = 0;
					if (cols_to_set[pt_idx] > 0 && skeleton.data[key - 1] == 128) // left
						skeleton.data[key - 1] = 255;
					if (cols_to_set[pt_idx] < skeleton.cols - 1 && skeleton.data[key + 1] == 128) // right
						skeleton.data[key + 1] = 255;
					if (rows_to_set[pt_idx] > 0 && skeleton.data[key - skeleton.cols] == 128) // up
						skeleton.data[key - skeleton.cols] = 255;
					if (rows_to_set[pt_idx] < skeleton.rows - 1 && skeleton.data[key + skeleton.cols] == 128) // down
						skeleton.data[key + skeleton.cols] = 255;
				}

				if ((niters++) >= max_iters) // we have done!
					break;
			}
		}

		skeleton = (skeleton != 0);
		return true;
	}
	void charsJugdeFeature(const cv::Mat& image, cv::Mat& features){
		cv::Mat img_bin;
		if (image.type() == CV_8UC3 || image.type() == CV_8UC4){
			cv::Mat grayImage;
			cv::cvtColor(image, img_bin, CV_RGB2GRAY);
		}
		else {
			img_bin = image.clone();
		}
		
		cv::Mat img_canny;
		cv::Canny(img_bin, img_canny, 100, 200, 3);
		std::vector<float> erFeatures(10, 0);
		erFeatures[0] = countNonZero(img_canny) / countNonZero(img_bin);

		cv::Mat  bw;
		cv::copyMakeBorder(img_bin, bw, 5, 5, 5, 5, cv::BORDER_CONSTANT, cv::Scalar(0));
		std::vector<std::vector<cv::Point> > contours0;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(bw, contours0, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		cv::RotatedRect rrect = cv::minAreaRect(contours0.at(0));
		erFeatures[3] = std::max(rrect.size.width, rrect.size.height) / std::min(rrect.size.width, rrect.size.height);//axial_ratio
		
		cv::Rect_<float> safeBoundRect;
		Locate::calcSafeRect(rrect, bw, safeBoundRect);
		cv::Mat bw_new;
		bw(safeBoundRect).copyTo(bw_new);
		cv::copyMakeBorder(bw_new, bw_new, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
		cv::Mat skeleton = cv::Mat::zeros(bw_new.size(), CV_8UC1);
		guo_hall_thinning(bw_new, skeleton);
		cv::Mat mask;
		skeleton(cv::Rect(5, 5, bw_new.cols - 10, bw_new.rows - 10)).copyTo(mask);
		bw_new(cv::Rect(5, 5, bw_new.cols - 10, bw_new.rows - 10)).copyTo(bw_new);
		cv::Scalar mean, std;
		cv::Mat tmp;
		cv::distanceTransform(bw_new, tmp, DIST_L1, 3);
		cv::meanStdDev(tmp, mean, std, mask);
		erFeatures[1]= mean[0];//stroke.mean
		erFeatures[2]= std[0];//stroke.width

		cv::Moments mu = cv::moments(contours0.at(0));
		double hu_moments[7];
		cv::HuMoments(mu, hu_moments);
		//erFeatures.push_back(hu_moments);
		std::vector<cv::Point> hull;
		cv::convexHull(contours0[0], hull);
		erFeatures[3] = (float)cv::contourArea(hull) / cv::contourArea(contours0[0]);//convex_hull_ratio
		std::vector<cv::Vec4i> cx;
		std::vector<int> hull_idx;
		//TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission
		//     if the region is very small because otherwise we'll loose all the convexities
		cv::approxPolyDP(cv::Mat(contours0[0]), contours0[0], (float)std::min(rrect.size.width, rrect.size.height) / 17, true);
		convexHull(contours0[0], hull_idx, false, false);
		if (hull_idx.size()>2)
		if (contours0[0].size()>3)
			convexityDefects(contours0[0], hull_idx, cx);
		erFeatures[4] = (int)cx.size();//convexities
		features = cv::Mat ( erFeatures);
	}

  bool getFeatureFromER(const cv::Mat& image, cv::Mat &features){
		cv::Mat img_bin;
		if (image.type() == CV_8UC3 || image.type() == CV_8UC4){
			cv::Mat grayImage;
			cv::cvtColor(image, img_bin, CV_RGB2GRAY);
		}
		else {
			img_bin = image.clone();
		}
		cv::Mat inv_img;
		cv::threshold(img_bin,inv_img, 100, 255, CV_THRESH_BINARY_INV);
		std::vector<cv::text::ERStat> erRegions;
		cv::Ptr<cv::text::ERFilter> er_filter = cv::text::createERFilterNM1(cv::text::loadDummyClassifier(), 128, 0.1f, 0.9f, 0.f, true);
		er_filter->run(inv_img, erRegions);
		float erFeatures[11] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		if (erRegions.size()>1 && (erRegions[1].rect != cv::Rect(0, 0, 24, 24))){
			cv::Mat roi = img_bin(erRegions[0].rect);
			cv::copyMakeBorder(img_bin, img_bin, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
			cv::Mat skeleton = cv::Mat::zeros(img_bin.size(), CV_8UC1);
			guo_hall_thinning(img_bin, skeleton);
			cv::Mat mask;
			skeleton(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(mask);
			img_bin(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(img_bin);
			cv::Scalar mean, std;
			cv::Mat tmp;
			cv::distanceTransform(img_bin, tmp, DIST_L1, 3);
			cv::meanStdDev(tmp, mean, std, mask);
			erFeatures[0] = mean[0];//stroke.mean
			erFeatures[1] = std[0];//stroke.width
			erFeatures[2] = 1.0*erRegions[1].area/erRegions[1].perimeter;//compectness
			erFeatures[3]=(erRegions[1].med_crossings);//穿越中值次数
			erFeatures[4] = (1 - erRegions[1].euler);//number of hole
			erFeatures[5] = (1.0*erRegions[1].rect.height) / erRegions[1].rect.width;//aspect ratio
			erFeatures[6] = erRegions[1].raw_moments[0];
			erFeatures[7] = erRegions[1].raw_moments[1];
			erFeatures[8] = erRegions[1].central_moments[0];
			erFeatures[9] = erRegions[1].central_moments[1];
			erFeatures[10] = erRegions[1].central_moments[2];
		}
       else
		    return false;
		features = cv::Mat(1, 11, CV_32FC1, erFeatures);
		//std::cout << features << std::endl;
		return true;
	}
	void svmTrain(){
		std::string xmlFile="../resources/train/charsJudge/charJudge.xml";
		std::string a = "../resources/train/charsJudge/";
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::LINEAR);//chi2
		svm->setDegree(10.0);//0.1
		// 1.4 bug fix: old 1.4 ver gamma is 1
		svm->setGamma(10.12);//0.1
		svm->setCoef0(1.0);//0.1
		svm->setC(25.63);//1
		svm->setNu(0.5);//0.1
		svm->setP(1.0);//0.1
		//svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));
		svm->setTermCriteria(cvTermCriteria(2, (int)1e7, 1e-6));
		std::vector<std::pair<std::string, SvmLabel>> train_file_list;
		std::vector<std::string> has_folder = { "has" };// "has1", "has3"};
		std::vector<std::string> no_folder = { "no" };//"no2", "no3"};
		
		std::vector<std::string> train_list;
		
		for (auto i = 0; i < has_folder.size(); i++){
			train_list = utils::getFiles(a + has_folder[i]);
			for (auto file : train_list)
				train_file_list.push_back(make_pair(file, kForward));
		}
		for (auto i = 0; i <no_folder.size(); i++){
			train_list = utils::getFiles(a + no_folder[i]);
			for (auto file : train_list)
				train_file_list.push_back(make_pair(file, kInverse));
		}
		std::random_shuffle(train_file_list.begin(), train_file_list.end());

		std::cout << train_file_list.size() << std::endl;

		cv::Mat samples;
		std::vector<int> responses;

		for (auto f : train_file_list) {
			auto image = cv::imread(f.first,0);
			if (!image.data) {
				fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
				continue;
			}
			
			// feature = getHistogram(image);
			cv::Mat img_bin = image.clone();
			cv::Mat inv_img;
			cv::threshold(img_bin, inv_img, 100, 255, CV_THRESH_BINARY_INV);
			std::vector<cv::text::ERStat> erRegions;
			cv::Ptr<cv::text::ERFilter> er_filter = cv::text::createERFilterNM1(cv::text::loadDummyClassifier(), 128, 0.0f, 1.0f, 0.0f, true);
			er_filter->run(inv_img, erRegions);
			er_filter->run(inv_img, erRegions);
			float erFeatures[20] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0,0,0};
			bool effect = true;
			if (erRegions.size() > 1){
				cv::Mat roi = img_bin(erRegions[0].rect);
				cv::copyMakeBorder(img_bin, img_bin, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
				cv::Mat skeleton = cv::Mat::zeros(img_bin.size(), CV_8UC1);
				guo_hall_thinning(img_bin, skeleton);
				cv::Mat mask;
				skeleton(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(mask);
				img_bin(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(img_bin);
				cv::Scalar mean, std;
				cv::Mat tmp;
				cv::distanceTransform(img_bin, tmp, DIST_L1, 3);
				cv::meanStdDev(tmp, mean, std, mask);
				erFeatures[0] = mean[0];//stroke.mean
				erFeatures[1] = std[0];//stroke.width
				erFeatures[2] = 1.0*erRegions[1].area; //compectness
				erFeatures[3] = 1.0* erRegions[1].perimeter;
				erFeatures[4] = (erRegions[1].med_crossings);//��Խ��ֵ����
				erFeatures[5] = (1 - erRegions[1].euler);//number of hole
				erFeatures[6] = 1.0*erRegions[1].rect.height; //aspect ratio
				erFeatures[7] = 1.0*erRegions[1].rect.width;
				erFeatures[8] = erRegions[1].rect.x + erRegions[1].rect.width / 2;
				erFeatures[9] = erRegions[1].rect.y + erRegions[1].rect.height / 2;
				erFeatures[10] = erRegions[1].raw_moments[0];
				erFeatures[11] = erRegions[1].raw_moments[1];
				erFeatures[12] = erRegions[1].central_moments[0];
				erFeatures[13] = erRegions[1].central_moments[1];
				erFeatures[14] = erRegions[1].central_moments[2];
				erFeatures[15] = 1.0*erRegions[1].rect.height / erRegions[1].rect.width;
				erFeatures[16] = 1.0*sqrt(erRegions[1].area) / erRegions[1].perimeter;
				erFeatures[17] = 1.0*erRegions[1].hole_area_ratio;
				erFeatures[18] = 1.0*erRegions[1].convex_hull_ratio;
				erFeatures[19] = 1.0*erRegions[1].num_inflexion_points;
			}
			else{
				effect = false;
				std::string imageName = f.first.c_str();
				imageName = Utils::getFileName(imageName, 1);
				cv::imwrite(a+"maybe2/" + imageName, image);
			}
			cv::Mat feature = cv::Mat(1, 20, CV_32FC1, erFeatures);
			//std::cout << features << std::endl;

			if (effect){
				samples.push_back(feature);
				responses.push_back(f.second);
			}
		}
		std::cout << responses.size() << std::endl;
		std::vector<double> scalesArray(40, 0);

		for (int i = 0; i < samples.cols; i++){
			cv::Mat colMat = samples.col(i);
			double minv, maxv;
			cv::minMaxLoc(colMat, &minv, &maxv, NULL, NULL);
			scalesArray[i * 2] = minv;
			scalesArray[i * 2 + 1] = maxv;
			cv::normalize(colMat, colMat, 1, 0, CV_MINMAX);
		}
        
		cv::FileStorage fsData("../resources/train/charsJudge/datafile.xml", cv::FileStorage::WRITE);
		fsData << "Samples" << samples;
		fsData << "Response" << responses;
		fsData << "Scales" << scalesArray;
		fsData.release();
		/*cv::Mat samples;
		std::vector<int> responses;
		cv::FileStorage fsData("../resources/train/charsJudge/datafile.xml", cv::FileStorage::READ);
		fsData["Samples"]>>samples;
		fsData["Response"] >>responses;
		fsData.release();*/
		auto train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
			responses);
		fprintf(stdout, ">> Training SVM model, please wait...\n");
		
		cv::ml::ParamGrid Cgrid(0.01, 500, 5);//(7, 200, 1.2);
	  //  cv::ml::ParamGrid Ggrid(0.2, 40, 3);//(0.03, 0.2, 1.3);
		cv::ml::ParamGrid grid(0, 0, 0);
		svm->trainAuto(train_data, 10,Cgrid,grid, grid, grid,grid, grid, true);
		//cv::ml::ParamGrid Cgrid(10, 640, 2);//(7, 200, 1.2);
		//cv::ml::ParamGrid Ggrid(0.3, 0.9, 1.2);//(0.03, 0.2, 1.3);
		//gridCSearch(samples, responses, Cgrid, Ggrid, xmlFile);
		//svm->train(train_data);

		fprintf(stdout, ">> Training done.");
		fprintf(stdout, ">> Saving model file...\n");
		svm->save(xmlFile);

		fprintf(stdout, ">> Your SVM Model was saved to %s\n", xmlFile.c_str());
		fprintf(stdout, ">> Testing...\n");
	}

	void svmTest(std::string tureFile, std::string falseFile, std::string xmlFile) {
		// 1.4 bug fix: old 1.4 ver there is no null judge
		// if (NULL == svm_)
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(xmlFile);

		std::vector<std::pair<std::string, SvmLabel >>test_file_list;

		auto has_file_test_list = utils::getFiles(tureFile);
		std::random_shuffle(has_file_test_list.begin(), has_file_test_list.end());

		auto no_file_test_list = utils::getFiles(falseFile);
		std::random_shuffle(no_file_test_list.begin(), no_file_test_list.end());


		for (auto file : has_file_test_list)
			test_file_list.push_back(make_pair(file, kForward));
		std::cout << test_file_list.size() << std::endl;
		for (auto file : no_file_test_list)
			test_file_list.push_back(make_pair(file, kInverse));

		int count_all = test_file_list.size();
		int ptrue_rtrue = 0;
		int ptrue_rfalse = 0;
		int pfalse_rtrue = 0;
		int pfalse_rfalse = 0;
		std::string imageName;

		std::vector<double> scales;
		cv::FileStorage fsData("../resources/train/charsJudge/datafile.xml", cv::FileStorage::READ);
		fsData["Scales"] >> scales;
		fsData.release();

		for (auto f : test_file_list) {
			auto image = cv::imread(f.first,0);
			if (!image.data) {
				fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
				continue;
			}			
			cv::Mat img_bin = image.clone();
			cv::Mat inv_img;
			cv::threshold(img_bin, inv_img, 100, 255, CV_THRESH_BINARY_INV);
			std::vector<cv::text::ERStat> erRegions;
			cv::Ptr<cv::text::ERFilter> er_filter = cv::text::createERFilterNM1(cv::text::loadDummyClassifier(), 128, 0.0f, 1.0f, 0.0f, true);
			er_filter->run(inv_img, erRegions);
			er_filter->run(inv_img, erRegions);
			float erFeatures[20] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0 };
			bool effect = true;
			int predict = kInverse;
			if (erRegions.size() > 1){
				cv::Mat roi = img_bin(erRegions[0].rect);
				cv::copyMakeBorder(img_bin, img_bin, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
				cv::Mat skeleton = cv::Mat::zeros(img_bin.size(), CV_8UC1);
				guo_hall_thinning(img_bin, skeleton);
				cv::Mat mask;
				skeleton(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(mask);
				img_bin(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(img_bin);
				cv::Scalar mean, std;
				cv::Mat tmp;
				cv::distanceTransform(img_bin, tmp, DIST_L1, 3);
				cv::meanStdDev(tmp, mean, std, mask);
				erFeatures[0] = mean[0];//stroke.mean
				erFeatures[1] = std[0];//stroke.width
				erFeatures[2] = 1.0*erRegions[1].area; //compectness
				erFeatures[3] = 1.0* erRegions[1].perimeter;
				erFeatures[4] = (erRegions[1].med_crossings);//��Խ��ֵ����
				erFeatures[5] = (1 - erRegions[1].euler);//number of hole
				erFeatures[6] = 1.0*erRegions[1].rect.height; //aspect ratio
				erFeatures[7] = 1.0*erRegions[1].rect.width;
				erFeatures[8] = erRegions[1].rect.x + erRegions[1].rect.width / 2;
				erFeatures[9] = erRegions[1].rect.y + erRegions[1].rect.height / 2;
				erFeatures[10] = erRegions[1].raw_moments[0];
				erFeatures[11] = erRegions[1].raw_moments[1];
				erFeatures[12] = erRegions[1].central_moments[0];
				erFeatures[13] = erRegions[1].central_moments[1];
				erFeatures[14] = erRegions[1].central_moments[2];
				erFeatures[15] = 1.0*erRegions[1].rect.height / erRegions[1].rect.width;
				erFeatures[16] = 1.0*sqrt(erRegions[1].area) / erRegions[1].perimeter;
				erFeatures[17] = 1.0*erRegions[1].hole_area_ratio;
				erFeatures[18] = 1.0*erRegions[1].convex_hull_ratio;
				erFeatures[19] = 1.0*erRegions[1].num_inflexion_points;
				for (int i = 0; i < 20; ++i){
					float minval = scales[2 * i];
					float maxval = scales[2 * i + 1];
					if (maxval>minval)
						erFeatures[i] = (erFeatures[i] - minval) / (maxval - minval);

				}
				cv::Mat feature = cv::Mat(1, 20, CV_32FC1, erFeatures);
				//std::cout << "feature: " << feature << std::endl;
				predict = svm->predict(feature);
				//std::cout << "predict: " << predict << std::endl;
			}
						
			auto real = f.second;
			if (predict == kForward && real == kForward) ptrue_rtrue++;
			if (predict == kForward && real == kInverse){
				ptrue_rfalse++;
				imageName = Utils::getFileName(f.first.c_str(), 1);
				cv::imwrite("../resources/train/charsJudge/pf/"+ imageName, image);
			}
			if (predict == kInverse && real == kForward) {
				pfalse_rtrue++;
				imageName = Utils::getFileName(f.first.c_str(), 1);
				cv::imwrite("../resources/train/charsJudge/nt/" + imageName, image);
			}
			if (predict == kInverse && real == kInverse) pfalse_rfalse++;
		}


		std::cout << "count_all: " << count_all << std::endl;
		std::cout << "ptrue_rtrue: " << ptrue_rtrue << std::endl;
		std::cout << "ptrue_rfalse: " << ptrue_rfalse << std::endl;
		std::cout << "pfalse_rtrue: " << pfalse_rtrue << std::endl;
		std::cout << "pfalse_rfalse: " << pfalse_rfalse << std::endl;

		double precise = 0;
		if (ptrue_rtrue + ptrue_rfalse != 0) {
			precise = 1.0*ptrue_rtrue / (ptrue_rtrue + ptrue_rfalse);
			std::cout << "precise: " << precise << std::endl;
		}
		else {
			std::cout << "precise: "
				<< "NA" << std::endl;
		}

		double recall = 0;
		if (ptrue_rtrue + pfalse_rtrue != 0) {
			recall = 1.0* ptrue_rtrue / (ptrue_rtrue + pfalse_rtrue);
			std::cout << "recall: " << recall << std::endl;
		}
		else {
			std::cout << "recall: "
				<< "NA" << std::endl;
		}

		double Fsocre = 0;
		if (precise + recall != 0) {
			Fsocre = 2.0 * (precise * recall) / (precise + recall);
			std::cout << "Fsocre: " << Fsocre << std::endl;
		}
		else {
			std::cout << "Fsocre: "
				<< "NA" << std::endl;
		}
	}
	
	void tag_data(std::string source_folder, std::string has_plate_folder,
		std::string no_plate_folder, std::string svm_model) {
		auto files = Utils::getFiles(source_folder);

		size_t size = files.size();
		if (0 == size) {
			std::cout << "No file found in " << source_folder << std::endl;
			return;
		}
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svm_model);
		std::string imageName;
		std::vector<double> scales;
		cv::FileStorage fsData("../resources/train/charsJudge/datafile.xml", cv::FileStorage::READ);
		fsData["Scales"] >> scales;
		fsData.release();
		for (auto f : files) {
			imageName = f.c_str();
			//std::cout << "--------" << imageName << std::endl;
			cv::Mat image = cv::imread(imageName, 0);
			if (!image.data) {
				fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.c_str());
				continue;
			}

			cv::Mat img_bin = image.clone();
			cv::Mat inv_img;
			cv::threshold(img_bin, inv_img, 100, 255, CV_THRESH_BINARY_INV);
			std::vector<cv::text::ERStat> erRegions;
			cv::Ptr<cv::text::ERFilter> er_filter = cv::text::createERFilterNM1(cv::text::loadDummyClassifier(), 128, 0.0f, 1.0f, 0.0f, true);
			er_filter->run(inv_img, erRegions);
			float erFeatures[17] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			bool effect = true;
			int predict = kInverse;
			if (erRegions.size() > 1){
				cv::Mat roi = img_bin(erRegions[0].rect);
				cv::copyMakeBorder(img_bin, img_bin, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
				cv::Mat skeleton = cv::Mat::zeros(img_bin.size(), CV_8UC1);
				guo_hall_thinning(img_bin, skeleton);
				cv::Mat mask;
				skeleton(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(mask);
				img_bin(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(img_bin);
				cv::Scalar mean, std;
				cv::Mat tmp;
				cv::distanceTransform(img_bin, tmp, DIST_L1, 3);
				cv::meanStdDev(tmp, mean, std, mask);
				erFeatures[0] = mean[0];//stroke.mean
				erFeatures[1] = std[0];//stroke.width
				erFeatures[2] = 1.0*erRegions[1].area; //compectness
				erFeatures[3] = 1.0* erRegions[1].perimeter;
				erFeatures[4] = (erRegions[1].med_crossings);//��Խ��ֵ����
				erFeatures[5] = (1 - erRegions[1].euler);//number of hole
				erFeatures[6] = 1.0*erRegions[1].rect.height; //aspect ratio
				erFeatures[7] = 1.0*erRegions[1].rect.width;
				erFeatures[8] = erRegions[1].rect.x + erRegions[1].rect.width / 2;
				erFeatures[9] = erRegions[1].rect.y + erRegions[1].rect.height / 2;
				erFeatures[10] = erRegions[1].raw_moments[0];
				erFeatures[11] = erRegions[1].raw_moments[1];
				erFeatures[12] = erRegions[1].central_moments[0];
				erFeatures[13] = erRegions[1].central_moments[1];
				erFeatures[14] = erRegions[1].central_moments[2];
				erFeatures[15] = 1.0*erRegions[1].rect.height / erRegions[1].rect.width;
				erFeatures[16] = 1.0*sqrt(erRegions[1].area) / erRegions[1].perimeter;

				for (int i = 0; i < 17; ++i){
					float minval = scales[2 * i];
					float maxval = scales[2 * i + 1];
					if (maxval>minval)
						erFeatures[i] = (erFeatures[i] - minval) / (maxval - minval);
				}
			}
			else
				effect = false;
			
			cv::Mat feature = cv::Mat(1, 17, CV_32FC1, erFeatures);
			if (effect = true){
			    predict = svm->predict(feature);
				imageName = Utils::getFileName(imageName, 1);
				if (predict == kForward)
					cv::imwrite(has_plate_folder + imageName, image);
				else
					cv::imwrite(no_plate_folder + imageName, image);
			}
			else{
				imageName = Utils::getFileName(imageName, 1);
				cv::imwrite(no_plate_folder + imageName, image);
			}
			//std::cout << "feature: " << feature << std::endl;
		}
	}

	void non_maximum_suppresion(const std::vector<std::vector<Point>> &all_contours, std::vector<cv::RotatedRect> &outRotatedRect, cv::Vec4f &charLine, std::vector<float>  &charSteps, std::vector<bool> &is1, cv::Mat& src) {

		std::vector<cv::RotatedRect> inRotatedRect;
		for (int i = 0; i < all_contours.size(); ++i) {
			cv::RotatedRect  new_rect = cv::minAreaRect(all_contours[i]);
			float angle = new_rect.angle;
			int area = new_rect.size.height*new_rect.size.width;
			float r = ((float)new_rect.size.height) / new_rect.size.width;
			cv::Size roi_size = new_rect.size;
			bool clockwise = true;
			if (r < 1) {
				angle = -(angle + 90);
				swap(roi_size.height, roi_size.width);
				r = 1 / r;
				clockwise = false;
			}
			if ((angle + 45 < 0) || (r > 8) || (area >src.rows*src.cols / 7) || area < 80)
				continue;
			inRotatedRect.push_back(new_rect);
		}
		/*cv::Mat temp = src.clone();
		std::cout << inRotatedRect.size() << std::endl;
		for (int i1 = 0; i1 < inRotatedRect.size(); ++i1) {
			cv::Point2f rect_points[4];
			inRotatedRect[i1].points(rect_points);
			for (int j = 0; j < 4; j++) {
				double a = i1 * 255 / inRotatedRect.size();
				cv::line(temp, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(255), 2, 8);
			}
		}
		cv::imwrite("temp/NMS2.jpg", temp);*/
		if (inRotatedRect.empty())
			return;
		std::sort(inRotatedRect.begin(), inRotatedRect.end(), [](const cv::RotatedRect &r1, const cv::RotatedRect &r2) {return r1.center.x < r2.center.x; });

		int marging = 0;
		std::vector<std::vector<cv::RotatedRect>> rrects_group;
		std::vector<cv::RotatedRect> rrects, midRR;
		for (int i = 0; i + 1 < inRotatedRect.size(); ++i) {
			float x1 = inRotatedRect[i].center.x;
			float x2 = inRotatedRect[i + 1].center.x;
			float x1_area = inRotatedRect[i].size.height*inRotatedRect[i].size.width;
			float x2_area = inRotatedRect[i + 1].size.height*inRotatedRect[i + 1].size.width;
			float x1_w = std::min(inRotatedRect[i].size.width, inRotatedRect[i].size.height);
			float x2_w = std::min(inRotatedRect[i + 1].size.width, inRotatedRect[i + 1].size.height);
			if (x1 < 8)
				continue;
			rrects.push_back(inRotatedRect[i]);

			if (x2 - x1 > 4) {
				rrects_group.push_back(rrects);
				rrects.clear();
			}
			else if ((x1_area>1.3*x2_area || x1_area<0.7 * x2_area) && abs(x1_w - x2_w)>6) {
				rrects_group.push_back(rrects);
				rrects.clear();
			}
			if (i + 2 == inRotatedRect.size()) {
				rrects.push_back(inRotatedRect[i + 1]);
				rrects_group.push_back(rrects);
			}
		}
		rrects.clear();
		for (int i = 0; i<rrects_group.size(); ++i) {
			std::sort(rrects_group[i].begin(), rrects_group[i].end(), [](const cv::RotatedRect &r1, const cv::RotatedRect &r2) {return ((r1.size.width*r1.size.height) >(r2.size.width*r2.size.height)); });
			rrects.push_back(rrects_group[i][0]);
		}
		/*temp = src.clone();
		std::cout << rrects.size() << std::endl;
		for (int i1 = 0; i1 < rrects.size(); ++i1) {
			cv::Point2f rect_points[4];
			rrects[i1].points(rect_points);
			for (int j = 0; j < 4; j++) {
				double a = i1 * 255 / rrects.size();
				cv::line(temp, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(255), 2, 8);
			}
		}
		cv::imwrite("temp/NMS.jpg", temp);*/
		
		int i2 = 0;
		for (; i2 + 2 < rrects.size(); ++i2) {
			midRR.push_back(rrects[i2]);
			float x1_w = std::min(rrects[i2].size.width, rrects[i2].size.height);
			float x1_r = rrects[i2].center.x + x1_w / 2;
			float x2_w = std::min(rrects[i2 + 1].size.width, rrects[i2 + 1].size.height);
			float x2_r = rrects[i2 + 1].center.x + x2_w / 2;
			float x2_l = rrects[i2 + 1].center.x - x2_w / 2;

			float x3_w = std::min(rrects[i2 + 2].size.width, rrects[i2 + 2].size.height);
			float x3_l = rrects[i2 + 2].center.x - x3_w / 2;

			float x1_area = rrects[i2].size.height*rrects[i2].size.width;
			float x2_area = rrects[i2 + 1].size.height*rrects[i2 + 1].size.width;
			float x3_area = rrects[i2 + 2].size.height*rrects[i2 + 2].size.width;
			float ratio = x2_area / x3_area;

			//std::cout << i2 << " is " << ratio;
			float x2_w_n = (5 * x2_w / 7);
			//int a = std::max(rrects[i].size.width, rrects[i + 1].size.width);
			if (x2_r - 3 < x3_l&& x2_l + 4 < x1_r) {
				//std::cout << "    Patern 4" << endl;
				if (x2_area > x1_area && x1_area<src.rows*src.cols/12) {
					midRR.pop_back();
					midRR.push_back(rrects[i2 + 1]);
				}
				else if (x2_area >src.rows*src.cols/7 &&x2_area>src.rows*src.cols / 12) {
					midRR.pop_back();
					midRR.push_back(rrects[i2 + 1]);
				}
				++i2;
			}
			else
			{
				if ((x3_l - x1_r) < x2_w_n) {
					if (ratio > 1) {
						++i2;
						//std::cout << "    Patern 2" << endl;
					}
					else {
						midRR.push_back(rrects[i2 + 1]);
						i2 += 2;
					//	std::cout << "   Patern 3" << endl;
					}
				}
				else {
					midRR.push_back(rrects[i2 + 1]);
					++i2;
					//std::cout << "   Patern 1" << std::endl;
				}
			}
		}
		for (; i2 < rrects.size(); ++i2)
			midRR.push_back(rrects[i2]);

		int outSize;
		if (midRR.empty())
			return;
		rrects_group.clear();
		rrects.clear();
		inRotatedRect.clear();
		inRotatedRect.assign(midRR.begin(), midRR.end());
		for (int i = 0; i < inRotatedRect.size(); ++i) {
			rrects.push_back(inRotatedRect[i]);
			float a1_height = std::max(rrects[0].size.height, rrects[0].size.width);
			float a1_angle, a2_angle;
			if (a1_height == rrects[0].size.height)
				a1_angle = rrects[0].angle;
			else a1_angle = -(rrects[0].angle + 90);
			if ((i + 1) < inRotatedRect.size()) {
				for (auto it = inRotatedRect.begin() + i + 1; it != inRotatedRect.end();) {
					float a2_height = std::max((*it).size.height, (*it).size.width);
					if (a2_height == (*it).size.height)
						a2_angle = (*it).angle;
					else  a2_angle = -((*it).angle + 90);

					if (std::abs(a1_height - a2_height) < 5 && std::abs(a1_angle - a2_angle) < 15 && std::abs((*it).center.y - rrects[0].center.y)<5) {
						rrects.push_back(*it);
						it = inRotatedRect.erase(it);
					}
					else  ++it;
				}
			}
			rrects_group.push_back(rrects);
			rrects.clear();
		}

		int max_index = 0;
		for (int i = 0; i < rrects_group.size(); ++i) {
			if (rrects_group[max_index].size()<rrects_group[i].size())
				max_index = i;
		}

		float avgAngle = 0, maxHeight = 0, avgY = 0;
		outSize = rrects_group[max_index].size();
		if (outSize == 0)
			return;
		for (int i = 0; i < outSize; ++i) {
			float height = std::max(rrects_group[max_index][i].size.height, rrects_group[max_index][i].size.width);
			if (maxHeight<height)
				maxHeight = height;
			if (rrects_group[max_index][i].size.height>rrects_group[max_index][i].size.width)
				avgAngle += rrects_group[max_index][i].angle;
			else
				avgAngle -= (rrects_group[max_index][i].angle + 90);
			avgY += rrects_group[max_index][i].center.y;
		}
		avgAngle = avgAngle / outSize;
		avgY = avgY / outSize;
		//std::cout << max_index << "  " << rrects_group[max_index].size() << "  " << avgAngle << "  " << maxHeight <<
		//	std::endl;

		bool kfirst = true;

		std::vector<cv::Point> charCenters;

		for (int i = 0; i < midRR.size(); ++i) {
			cv::Size new_size = midRR[i].size;
			cv::Point new_center = midRR[i].center;
			float new_angle = midRR[i].angle;
			float a_height = std::max(midRR[i].size.height, midRR[i].size.width);
			float a_width = std::min(midRR[i].size.width, midRR[i].size.height);
			float a_angle;
			bool MN = (midRR[i].size.height == a_height);
			a_angle = midRR[i].angle;
			if (!MN)
				a_angle = -(a_angle + 90);
			if (a_height + 3 < maxHeight) {
				new_center = cv::Point(midRR[i].center.x, avgY);
				if (MN)
					new_size = cv::Size(a_width, maxHeight);
				else
					new_size = cv::Size(maxHeight, a_width);
			}
			if ((a_height - 4 > maxHeight) && (midRR[i].center.x<30 || midRR[i].center.x>128))
				continue;
			Point2f i_points[4];
			midRR[i].points(i_points);
			float L = min(min(min(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			float R = max(max(max(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			if (abs(a_angle - avgAngle) > 8)
			{
				new_center.x = (L + R) / 2;
				if (MN)
					new_angle = avgAngle;
				else
					new_angle = -(avgAngle + 90);
				//new_center.x = new_center.x - 3;
			}
			if (a_height / a_width > 3) {
				is1.push_back(true);
				if (new_size.height >= new_size.width)
					new_size.width *= 2.8;
				else
					new_size.height *= 2.8;
			}
			else
				is1.push_back(false);
			if (kfirst) {
				if (MN) {
					new_size.height = new_size.height*1.2;
					new_size.width = new_size.height*0.6;

				}
				else {
					new_size.width = new_size.width*1.2;
					new_size.height = new_size.width*0.6;
				}
				kfirst = false;
			}
			cv::RotatedRect new_rect(new_center, new_size, new_angle);
			outRotatedRect.push_back(new_rect);
			charCenters.push_back(new_center);
		}

		//std::cout << rrects_group.size() << std::endl;
	/*	for (; i2 + 2 < rrects.size(); ++i2) {
			midRR.push_back(rrects[i2]);
			float x1_w = std::min(rrects[i2].size.width, rrects[i2].size.height);
			float x1_r = rrects[i2].center.x + x1_w / 2;
			float x2_w = std::min(rrects[i2 + 1].size.width, rrects[i2 + 1].size.height);
			float x2_r = rrects[i2 + 1].center.x + x2_w / 2;
			float x2_l = rrects[i2 + 1].center.x - x2_w / 2;

			float x3_w = std::min(rrects[i2 + 2].size.width, rrects[i2 + 2].size.height);
			float x3_l = rrects[i2 + 2].center.x - x3_w / 2;

			float x1_area = rrects[i2].size.height*rrects[i2].size.width;
			float x2_area = rrects[i2 + 1].size.height*rrects[i2 + 1].size.width;
			float x3_area = rrects[i2 + 2].size.height*rrects[i2 + 2].size.width;
			float ratio = x2_area / x3_area;

			float x2_w_n = (5 * x2_w / 7);
			if (x2_r - 3 < x3_l&& x2_l + 4 < x1_r) {
				if (x2_area > x1_area) {
					midRR.pop_back();
					midRR.push_back(rrects[i2 + 1]);
				}
				++i2;
			}
			else
			{
				if ((x3_l - x1_r) < x2_w_n) {
					if (ratio > 1) 
						++i2;
					else {
						midRR.push_back(rrects[i2 + 1]);
						i2 += 2;
					}
				}
				else {
					midRR.push_back(rrects[i2 + 1]);
					++i2;
				}
			}
		}
		for (; i2 < rrects.size(); ++i2)
			midRR.push_back(rrects[i2]);

		int outSize;
		if (midRR.empty())
			return;
		rrects_group.clear();
		rrects.clear();
		inRotatedRect.clear();
		inRotatedRect.assign(midRR.begin(), midRR.end());
		for (int i = 0; i < inRotatedRect.size(); ++i) {
			rrects.push_back(inRotatedRect[i]);
			float a1_height = std::max(rrects[0].size.height, rrects[0].size.width);
			float a1_angle, a2_angle;
			if (a1_height == rrects[0].size.height)
				a1_angle = rrects[0].angle;
			else a1_angle = -(rrects[0].angle + 90);
			if ((i + 1) < inRotatedRect.size()) {
				for (auto it = inRotatedRect.begin() + i + 1; it != inRotatedRect.end();) {
					float a2_height = std::max((*it).size.height, (*it).size.width);
					if (a2_height == (*it).size.height)
						a2_angle = (*it).angle;
					else  a2_angle = -((*it).angle + 90);

					if (std::abs(a1_height - a2_height) < 5 && std::abs(a1_angle - a2_angle) < 15 && std::abs((*it).center.y - rrects[0].center.y)<5) {
						rrects.push_back(*it);
						it = inRotatedRect.erase(it);
					}
					else  ++it;
				}
			}
			rrects_group.push_back(rrects);
			rrects.clear();
		}

		int max_index = 0;
		for (int i = 0; i < rrects_group.size(); ++i) {
			if (rrects_group[max_index].size()<rrects_group[i].size())
				max_index = i;
		}

		float avgAngle = 0, maxHeight = 0, avgY = 0;
		outSize = rrects_group[max_index].size();
		if (outSize == 0)
			return;
		for (int i = 0; i < outSize; ++i) {
			float height = std::max(rrects_group[max_index][i].size.height, rrects_group[max_index][i].size.width);
			if (maxHeight<height)
				maxHeight = height;
			if (rrects_group[max_index][i].size.height>rrects_group[max_index][i].size.width)
				avgAngle += rrects_group[max_index][i].angle;
			else
				avgAngle -= (rrects_group[max_index][i].angle + 90);
			avgY += rrects_group[max_index][i].center.y;
		}
		avgAngle = avgAngle / outSize;
		avgY = avgY / outSize;

		bool kfirst = true;

		std::vector<cv::Point> charCenters;

		for (int i = 0; i < midRR.size(); ++i) {
			cv::Size new_size = midRR[i].size;
			cv::Point new_center = midRR[i].center;
			float new_angle = midRR[i].angle;
			float a_height = std::max(midRR[i].size.height, midRR[i].size.width);
			float a_width = std::min(midRR[i].size.width, midRR[i].size.height);
			float a_angle=new_angle;
			bool MN = (midRR[i].size.height == a_height);
			if (!MN)
				a_angle = -(a_angle + 90);
			if (a_height + 3 < maxHeight) {
				new_center = cv::Point(midRR[i].center.x, avgY);
				if (MN)
					new_size = cv::Size(a_width, maxHeight);
				else
					new_size = cv::Size(maxHeight, a_width);
			}
			if ((a_height - 4 > maxHeight) && (midRR[i].center.x<30 || midRR[i].center.x>128))
				continue;
			Point2f i_points[4];
			midRR[i].points(i_points);
			float L = min(min(min(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			float R = max(max(max(i_points[1].x, i_points[2].x), i_points[3].x), i_points[0].x);
			if (abs(a_angle - avgAngle) > 8)
			{
				new_center.x = (L + R) / 2;
				if (MN)
					new_angle = avgAngle;
				else
					new_angle = -(avgAngle + 90);
				//new_center.x = new_center.x - 3;
			}
			if (a_height / a_width > 3) {
				is1.push_back(true);
				if (new_size.height >= new_size.width)
					new_size.width *= 2.8;
				else
					new_size.height *= 2.8;
			}
			else
				is1.push_back(false);
			if (kfirst) {
				if (MN) {
					new_size.height = new_size.height*1.2;
					new_size.width = new_size.height*0.6;

				}
				else {
					new_size.width = new_size.width*1.2;
					new_size.height = new_size.width*0.6;
				}
				kfirst = false;
			}
			cv::RotatedRect new_rect(new_center, new_size, new_angle);
			outRotatedRect.push_back(new_rect);
			charCenters.push_back(new_center);
		}
		*/

		int imm = 0;
		std::vector<bool>::iterator it2 = is1.begin();
		
		if (charCenters.size() > 1) {
			cv::fitLine(cv::Mat(charCenters), charLine, CV_DIST_L2, 0, 0.01, 0.01);
			for (int i = 0; i + 1 < charCenters.size(); ++i) {
				/*float step = std::sqrtf(std::powf((charCenters[i + 1].x - charCenters[i].x), 2) + std::powf((charCenters[i + 1].y - charCenters[i].y), 2));*/
				float step = abs(charCenters[i + 1].x - charCenters[i].x);
				charSteps.push_back(step);
			}
		}
	}
	bool get_charMat_feature(const cv::Mat &image, const std::vector<double> &scales, cv::Mat& featureout) {
		cv::Mat img_bin = image.clone();
		cv::Mat inv_img;
		cv::threshold(img_bin, inv_img, 100, 255, CV_THRESH_BINARY_INV);
		std::vector<cv::text::ERStat> erRegions;
		cv::Ptr<cv::text::ERFilter> er_filter = cv::text::createERFilterNM1(cv::text::loadDummyClassifier(), 128, 0.0f, 1.0f, 0.0f, true);
		er_filter->run(inv_img, erRegions);
		er_filter->run(inv_img, erRegions);
		float erFeatures[20] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
		bool effect = true;

		if (erRegions.size() > 1) {
			cv::Mat roi = img_bin(erRegions[0].rect);
			cv::copyMakeBorder(img_bin, img_bin, 5, 5, 5, 5, BORDER_CONSTANT, Scalar(0));
			cv::Mat skeleton = cv::Mat::zeros(img_bin.size(), CV_8UC1);
			guo_hall_thinning(img_bin, skeleton);
			cv::Mat mask;
			skeleton(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(mask);
			img_bin(cv::Rect(5, 5, img_bin.cols - 10, img_bin.rows - 10)).copyTo(img_bin);
			cv::Scalar mean, std;
			cv::Mat tmp;
			cv::distanceTransform(img_bin, tmp, DIST_L1, 3);
			cv::meanStdDev(tmp, mean, std, mask);
			erFeatures[0] = mean[0];//stroke.mean
			erFeatures[1] = std[0];//stroke.width
			erFeatures[2] = 1.0*erRegions[1].area; //compectness
			erFeatures[3] = 1.0* erRegions[1].perimeter;
			erFeatures[4] = (erRegions[1].med_crossings);//��Խ��ֵ����
			erFeatures[5] = (1 - erRegions[1].euler);//number of hole
			erFeatures[6] = 1.0*erRegions[1].rect.height; //aspect ratio
			erFeatures[7] = 1.0*erRegions[1].rect.width;
			erFeatures[8] = erRegions[1].rect.x + erRegions[1].rect.width / 2;
			erFeatures[9] = erRegions[1].rect.y + erRegions[1].rect.height / 2;
			erFeatures[10] = erRegions[1].raw_moments[0];
			erFeatures[11] = erRegions[1].raw_moments[1];
			erFeatures[12] = erRegions[1].central_moments[0];
			erFeatures[13] = erRegions[1].central_moments[1];
			erFeatures[14] = erRegions[1].central_moments[2];
			erFeatures[15] = 1.0*erRegions[1].rect.height / erRegions[1].rect.width;
			erFeatures[16] = 1.0*sqrt(erRegions[1].area) / erRegions[1].perimeter;
			erFeatures[17] = 1.0*erRegions[1].hole_area_ratio;
			erFeatures[18] = 1.0*erRegions[1].convex_hull_ratio;
			erFeatures[19] = 1.0*erRegions[1].num_inflexion_points;
			for (int i = 0; i < 20; ++i) {
				float minval = scales[2 * i];
				float maxval = scales[2 * i + 1];
				if (maxval>minval)
					erFeatures[i] = (erFeatures[i] - minval) / (maxval - minval);

			}
			cv::Mat feature = cv::Mat(1, 20, CV_32FC1, erFeatures);
			featureout = feature.clone();
			return true;
		}
		else
			return false;
	}

	bool get_features_image(const cv::Mat &src, const std::vector<double> &scales, const cv::RotatedRect &rrect,cv::Mat &features,cv::Mat &charMat,bool &isChar){
		isChar = true;
		float angle = rrect.angle;
		float r = ((float)rrect.size.height) / rrect.size.width;
		cv::Size roi_size = rrect.size;
		bool clockwise = true;
		if (r < 1) {
			angle = -(angle + 90);
			swap(roi_size.height, roi_size.width);
			r = 1 / r;
			clockwise = false;
		}
		roi_size.width *= (int)1.4;
		roi_size.height *= (int)1.4;
		cv::Rect_<float> safeBoundRect;
		bool isFormRect = Locate::calcSafeBigRect(rrect, src, safeBoundRect);
		cv::Point center = rrect.center - safeBoundRect.tl();
		cv::Mat rotated_char;
		float rotate_angle = (clockwise) ? angle : (-angle);
		if (!Locate::rotation(src(safeBoundRect), rotated_char, roi_size, center, rotate_angle)) 
			return false;

		float ratio = (float)rotated_char.cols / (float)rotated_char.rows;
		cv::resize(rotated_char, rotated_char, cv::Size(int(24 * ratio), 24), 0, CV_INTER_AREA);
		cv::threshold(rotated_char, rotated_char, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		int expendW = (24 - rotated_char.cols) / 2;
		cv::Mat out(24, 24, CV_8UC1, cv::Scalar(0));
		cv::Mat outRoi = out(cv::Rect(expendW, 0, rotated_char.cols, rotated_char.rows));
		rotated_char.copyTo(outRoi);
		charMat = out.clone();
		isChar = get_charMat_feature(out, scales, features);
		return true;
	}
	void plate_candidate_anaysis(std::vector<CPlate> &cplate_candis, std::vector<int> &true_plate_index) {
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("../resources/model/char_judge/charJudge.xml");
		std::vector<double> scales;
		cv::FileStorage fsData("../resources/model/char_judge/datafile.xml", cv::FileStorage::READ);
		fsData["Scales"] >> scales;
		fsData.release();
		for (int cplate_index = 0; cplate_index < cplate_candis.size();++cplate_index) {
			cv::Mat image = cplate_candis[cplate_index].plateMat;
			/*if (image.size() != cv::Size(153, 48))
				continue;*/
			cv::Mat grey;
			if (image.type() != CV_8UC1)
				cvtColor(image, grey, COLOR_RGB2GRAY);
			else grey = image.clone();
			resize(grey,grey, Size(153, 48), 1);
			std::vector<std::vector<Point>>all_contours;
			std::vector<cv::Rect> all_boxes;
			cv::Ptr<cv::MSER2> mser;
			mser = cv::MSER2::create();
			mser->detectBrightRegions(grey, all_contours, all_boxes);
			std::vector<cv::RotatedRect> rrects;
			if (!all_contours.empty()) {
				cv::Vec4f charLine;
				std::vector<float> charSteps;
				std::vector<bool> is1;
				non_maximum_suppresion(all_contours, rrects, charLine, charSteps, is1, image);
				if (rrects.size() < 2) {
					cplate_candis[cplate_index].isPlate = false;
					continue;
				}
				std::vector<bool> isChars;
				std::vector<cv::Mat> charMats;
				for (int charJudgeI1 = 0; charJudgeI1 < rrects.size(); ++charJudgeI1) {
					cv::Mat features;
					cv::Mat charMat;
					bool anm;
					if(!get_features_image(grey, scales, rrects[charJudgeI1], features, charMat, anm))
						continue;
					charMats.push_back(charMat);
					if (anm == true)
						isChars.push_back(svm->predict(features));
					else isChars.push_back(false);
				}
				// = GroupChar_judge(rrects, image);
				int charNum = 0;
				for (int i = 0; i<isChars.size(); ++i) {
					if ((isChars[i] || is1[i]) == 1)//is 1 or char
						charNum++;
				}//int charNum = countNonZero(isChars);
				if (charNum <2) {
					cplate_candis[cplate_index].isPlate = false;
					continue;
				}
				bool firstChinese = false;
				if ((charNum ==6 && isChars[0] + is1[0] == 0)||charNum==7) {
					firstChinese = true;
					cplate_candis[cplate_index].isPlate = true;
					cplate_candis[cplate_index].charMats = charMats;
					cplate_candis[cplate_index].is1 = is1;
					true_plate_index.push_back(cplate_index);
					continue;
				}
				//	temp = image.clone();

				int firstI = 0, lastI = rrects.size() - 1;
				while (isChars[firstI] +is1[firstI]==0) {
					++firstI;
				}
				while (isChars[lastI]+is1[lastI]==0) {
					--lastI;
				}
				/*if (isChars[0] + is1[0] == 0 && firstI == 1)
				firstChinese = true;*/
				bool hasLEndpoint = false, hasREndpoint = false;
				cv::RotatedRect chineseR;
				cv::Mat chineseMat;
				if (firstI != 0) {
					hasLEndpoint = true;//not 1 and char
					chineseR = rrects[firstI - 1];
					chineseMat = charMats[firstI - 1];
				}
				if (lastI != (rrects.size() - 1))
					hasREndpoint = true;
				cv::RotatedRect leftER = rrects[firstI];
				cv::RotatedRect rightER = rrects[lastI];
				std::vector<cv::RotatedRect> rightRects;
				std::vector<cv::Mat> rightcharMats;
				std::vector<cv::RotatedRect> OverallRects;
				std::vector<cv::Mat> OverallcharMats;
				std::vector<cv::RotatedRect> leftRects;
				std::vector<cv::Mat> leftcharMats;
				std::vector<bool> Overall_is1;
				std::vector<bool> expand_is1;
				float k = charLine[1] / charLine[0];
				float avgStep = 0;

				{
					float center_x, x1_w, x1_h, x1_l, ang1;
					center_x = leftER.center.x;
					x1_w = std::min((leftER).size.width, (leftER).size.height) / 2;
					x1_h = std::max((leftER).size.width, (leftER).size.height);
					x1_l = (leftER).center.x - x1_w;
					ang1 = std::max((leftER).angle, -(leftER).angle - 90);
					bool go_next = false;
					int stepx = 2.6*x1_w;
					float stopx = 0;
					if (hasLEndpoint)
						stopx = cv::max(rrects[firstI - 1].center.x + rrects[firstI - 1].size.width / 2, (float)0);
					while (center_x - stepx > stopx*1.2) {
						cv::RotatedRect leftNew(cv::Point(center_x - stepx, k*(center_x - stepx - charLine[2]) + charLine[3]), cv::Size(x1_w*2.4, x1_h*1.2), ang1);
						cv::Mat features;
						cv::Mat charMat;
						bool anm;
						if (!get_features_image(grey, scales, leftNew, features, charMat, anm))
							continue;
						bool insBs;	
						if (anm == true)
							insBs=svm->predict(features);
						else insBs=false;
						if (insBs) {
							leftRects.push_back(leftNew);
							leftcharMats.push_back(charMat);
							center_x = center_x - stepx;
							stepx = 2.6*x1_w;
							
						}
						else {
							stepx *= 1.3;
							if (stepx>5 * x1_w)
								break;
							//expandRrects.push_back(leftNew);
						}
					}
					stepx = 2.6*x1_w;
					if (center_x - stepx>2.4*x1_w) {
						chineseR.center = cv::Point(center_x - stepx, k*(center_x - stepx - charLine[2]) + charLine[3]); chineseR.size = cv::Size(x1_w*2.6, x1_h*1.4);
						chineseR.angle = ang1;
						cv::Mat features;
						cv::Mat charMat;
						bool anm;
						if (!get_features_image(grey, scales, chineseR, features, chineseMat, anm))
							continue;
						hasLEndpoint = true;
					}
				}
				{
					float center_x, x1_w, x1_h, ang1;
					center_x = rightER.center.x;
					x1_w = std::min((rightER).size.width, (rightER).size.height) / 2;
					x1_h = std::max((rightER).size.width, (rightER).size.height);
					ang1 = std::max((rightER).angle, -(rightER).angle - 90);
					bool go_next = false;
					int stepx = 2.6*x1_w;
					float stopx = (float)grey.cols - 1;
					if (hasREndpoint)
						stopx = cv::min(rrects[lastI + 1].center.x - rrects[lastI + 1].size.width / 2, (float)grey.cols - 1);
					while (center_x + stepx*1.2  <stopx) {
						cv::RotatedRect rightNew(cv::Point(center_x + stepx, k*(center_x + stepx - charLine[2]) + charLine[3]), cv::Size(x1_w*2.4, x1_h*1.2), ang1);
						cv::Mat features;
						cv::Mat charMat;
						bool anm;
						if (!get_features_image(grey, scales, rightNew, features, charMat, anm))
							continue;
						bool insBs;
						if (anm == true)
							insBs = svm->predict(features);
						else insBs = false;
						if (insBs) {
							rightRects.push_back(rightNew);
							rightcharMats.push_back(charMat);
							center_x = center_x + stepx;
							stepx = 2.6*x1_w;
						}
						else {
							stepx *= 1.3;
							if (stepx>5 * x1_w)
								break;
							//expandRrects.push_back(leftNew);
						}
					}
				}
				std::vector<cv::RotatedRect> expandRrects;
				std::vector<cv::Mat> expandcharMats;
				if (charNum>1)
				{
					for (int i2 = 0; i2 < rrects.size(); ++i2) {
						if (isChars[i2] == false && is1[i2] == false)
							continue;
						expandRrects.push_back(rrects[i2]);
						expandcharMats.push_back(charMats[i2]);
						expand_is1.push_back(is1[i2]);
						bool cheakGap = false;
						for (int j = i2 + 1; j < rrects.size(); ++j) {
							if (isChars[j] == false && is1[j] == false)
								continue;
							expandRrects.push_back(rrects[j]);
							expandcharMats.push_back(charMats[j]);
							expand_is1.push_back(is1[j]);
							cheakGap = true;
							break;
						}
						bool go_next = false;
						cv::RotatedRect insertR;
						while (cheakGap || go_next) {
							auto itend1 = expandRrects.rbegin();
							auto itend1_f = expandcharMats.rbegin();
							auto itend1_1 = expand_is1.rbegin();
							float x2_w = std::min((*itend1).size.width, (*itend1).size.height) / 2;
							float x2_h = std::max((*itend1).size.width, (*itend1).size.height);
							float x2_r = (*itend1).center.x + x2_w;
							float x2_l = (*itend1).center.x - x2_w;
							float ang2 = std::max((*itend1).angle, -(*itend1).angle - 90);

							float x1_w, x1_h, x1_r, x1_l, ang1;
							if (cheakGap) {
								auto itend2 = expandRrects.rbegin() + 1;
								x1_w = std::min((*itend2).size.width, (*itend2).size.height) / 2;
								x1_h = std::max((*itend2).size.width, (*itend2).size.height);
								x1_r = (*itend2).center.x + x1_w;
								x1_l = (*itend2).center.x - x1_w;
								ang1 = std::max((*itend2).angle, -(*itend2).angle - 90);
								cheakGap = false;
							}
							else {
								x1_w = std::min((insertR).size.width, (insertR).size.height) / 2;
								x1_h = std::max((insertR).size.width, (insertR).size.height);
								x1_r = (insertR).center.x + x1_w;
								x1_l = (insertR).center.x - x1_w;
								ang1 = std::max((insertR).angle, -(insertR).angle - 90);
							}
							if (std::abs(x1_r - x2_l) > 1.1*(x1_w + x2_w)) {
								cv::Point insertC;
								if (std::abs(x1_r - x2_l) > 1.5*(x1_w + x2_w) && std::abs(x1_r - x2_l) <2.3 * (x1_w + x2_w)) {
									insertC.x = x2_l - 1.3*x2_w;
									go_next = false;
								}
								else if (std::abs(x1_r - x2_l) > 2.3*(x1_w + x2_w)) {
									insertC.x = x1_r + 1.3*x2_w;
									go_next = true;
								}
								else {
									insertC.x = x1_r + 1.3*x2_w;
									go_next = false;
								}
								insertC.y = k*(insertC.x - charLine[2]) + charLine[3];

								float insertA = (ang1 + ang2) / 2;
								cv::Size insertS;
								insertS.height = (x2_h + x1_h) / 2;
								insertS.width = 2 * std::max(x1_w, x2_w);
								insertR.center = insertC;
								insertR.size = insertS;
								insertR.angle = insertA;
								cv::Mat features;
								cv::Mat charMat;
								bool anm;
								if (!get_features_image(grey, scales, insertR, features, charMat, anm))
									continue;
								bool insBs;
								if (anm == true)
									insBs = svm->predict(features);
								else insBs = false;
								if (insBs) {
									cv::RotatedRect failOne = *itend1;//((*itend1).center, (*itend1).size, (*itend1).angle);
									cv::Mat failMat = *itend1_f;
									bool fail_is1 = *itend1_1;

									expandRrects.pop_back();
									expandcharMats.pop_back();
									expand_is1.pop_back();

									expandRrects.push_back(insertR);
									expandRrects.push_back(failOne);
									expandcharMats.push_back(charMat);
									expandcharMats.push_back(failMat);
									expand_is1.push_back(false);
									expand_is1.push_back(fail_is1);
								}
							}
							else
								go_next = false;
						}
					}
					
					if (hasLEndpoint) {
						OverallRects.push_back(chineseR);
						OverallcharMats.push_back(chineseMat);
						Overall_is1.push_back(false);
						for (int i = leftRects.size() - 1; i >= 0; --i) {
							OverallRects.push_back(leftRects[i]);
							OverallcharMats.push_back(leftcharMats[i]);
							Overall_is1.push_back(false);
						}
						for (int i = 0; i < expandRrects.size(); ++i) {
							OverallRects.push_back(expandRrects[i]);
							OverallcharMats.push_back(expandcharMats[i]);
							Overall_is1.push_back(expand_is1[i]);
						}
						for (int i = 0; i < rightRects.size(); ++i) {
							OverallRects.push_back(rightRects[i]);
							OverallcharMats.push_back(rightcharMats[i]);
							Overall_is1.push_back(false);
						}
						if (OverallRects.size() == 7) {
							cplate_candis[cplate_index].isPlate = true;
							cplate_candis[cplate_index].charMats = OverallcharMats;
							cplate_candis[cplate_index].is1 = Overall_is1;
							true_plate_index.push_back(cplate_index);
						}
						else
							cplate_candis[cplate_index].isPlate = false;
					}
					else
						cplate_candis[cplate_index].isPlate = false;
				}
			}
		}
		svm.release();
		return;
	}
}

#endif