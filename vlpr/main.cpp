//#include "mser2.hpp"
//#include"creatDataset.h"
#include"chars_judge.hpp"
#include "characterR.hpp"
#include "cnn.hpp"
//#include "platelocate.hpp"
#define Debug

#ifdef Debug
#include <iostream>
#include <time.h>
#endif

using namespace cv;

int main(int argc, char* argv[]){
#ifdef Debug
	clock_t start, finish;
	double totaltime;
	start = clock();
#endif
	/*Mat src=imread("./temp/data11/1.jpg");
	vector<Mat> candidates;
	Locate::sobelDet(src,candidates);*/
     //creatDataSet("testCandi");//locate plate candidates;
	//mser1Test("ppl");
	//creatMserSet("ppl");
	//const std::string a="../resources/image/sense/";
	//const std::string a = "../resources/train/charsJudge/";
	//Mat src = imread(a + "0.jpg",0);
	//threshold(src, src, 100, 255, CV_THRESH_BINARY_INV);
	//Mat features;
	//charsJudge::getFeatureFromER(src, features);
	//charsJudge::svmTrain();
	//charsJudge::svmTest(a + "has/test", a + "no/test", a + "svm.xml");
	//charsJudge::tag_data(a + "test", a + "testhas/", a + "testno/", a + "svm.xml");
    // charsJudge::creatLibsvmData( "libData.txt");       
	// calculate("ppl");
	//creatCharSet();
	//calcuResize();
	//calcuResize2();
	//charsJudge::svmTrain();
	//charsJudge::svmTest(a + "testhas", a + "testno", a + "charJudge.xml");
	//charsJudge::tag_data(a + "testno", a + "nf/", a + "pf/", a + "charJudge.xml");
	//numCharRecognition();
	//charRTest();
	//chinese_ann_train();
	//chinese_ann_charRTest();
	//create_CSet();
	/*cv::Mat img=cv::imread("wan.jpg", 0);
	std::string chinese_pin;
	cnn_chinesesR(img, chinese_pin);
	std::shared_ptr<Kv> kv = std::shared_ptr<Kv>(new Kv);
	kv->load("../resources/train/chinese/province_mapping");
	std::string province = kv->get(chinese_pin);
	std::cout << "The character is " << province << std::endl;
	
	std::vector < CPlate> cplate_candis;
	std::vector<int> true_plate_index;
	CPlate cplate;
	cplate.plateMat = cv::imread("2.jpg", 0);
	cplate_candis.push_back(cplate);
	cplate.plateMat = cv::imread("3.jpg", 0);
	cplate_candis.push_back(cplate);
	cplate.plateMat = cv::imread("4.jpg", 0);
	cplate_candis.push_back(cplate);
	cplate.plateMat = cv::imread("5.jpg", 0);
	cplate_candis.push_back(cplate);
	charsJudge::plate_candidate_anaysis(cplate_candis, true_plate_index);
	std::cout << true_plate_index.size() << std::endl;*/
	/*std::cout << cplate_candis2.size() << std::endl;
	char file[100];
	std::string fileaa = "temp/";
	for (int j = 0; j < cplate_candis2.size(); ++j) {
	sprintf_s(file, "candi%d.jpg", j);
	cv::imwrite(fileaa + file, cplate_candis2[j].plateMat);
	}*/
	string src_path = "../resources/image/lifeCar/download";// D:\vlpr\resources\image\lifeCar\my - iphone
	auto files = Utils::getFiles(src_path);
	size_t size = files.size();
	if (0 == size) {
		std::cout << "No File Found!!" << std::endl;
		return -1;
	}
	else {
		std::cout << "Begin to Test the overall function!" << std::endl;
		cv::Mat img;
		std::string imageName;
		int locateFailure = 0;
		for (size_t i = 0; i < size; i++) {
			imageName = files[i].c_str();
			std::cout << "--------" << imageName << std::endl;
			img = cv::imread(imageName);
			std::vector < CPlate> cplate_candis;
			std::vector<int> true_plate_index;
			std::vector<bool> methods = { false,true,false,false };
			Locate::CplateDetect(img, cplate_candis, methods);
			cv::imwrite("Debug_plate.jpg", cplate_candis[0].plateMat);
			charsJudge::plate_candidate_anaysis(cplate_candis, true_plate_index);
		
			if (true_plate_index.empty()) {
				++locateFailure;
				std::cout << "We don't find any plate in this picture!" << std::endl;
				std::cout << "Maybe you can choose other method to locate your plate......" << std::endl;
			}
			else {
				for (int i = 0; i < true_plate_index.size(); ++i) {
					std::vector<std::string> num_alpha;
					cv::Mat img = cplate_candis[true_plate_index[0]].charMats[0];
					copyMakeBorder(img, img, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0));
					std::string province;
					cnn_chinesesR(img, province);
					num_alpha.push_back(province);
					num_alpha_recog(cplate_candis[true_plate_index[i]], num_alpha);
					std::cout << "The recognition result is: ";
					for (int numAl = 0; numAl < num_alpha.size(); ++numAl) {
						std::cout << num_alpha[numAl];
					}
					std::cout << std::endl;
				}
			}
		}
		std::cout << "LocateFailure is " << locateFailure << " which means " << locateFailure / (size*1.0) * 100 << "%" << " Locate rate" << std::endl;
	}
#ifdef Debug
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "总的运行时间为：  " << totaltime <<"秒"<< std::endl;
#endif
	system("Pause");
	return 0;
}