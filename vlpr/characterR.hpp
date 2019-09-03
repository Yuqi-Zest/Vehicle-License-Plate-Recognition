#ifndef CHARACTERR_HPP
#define CHARACTERR_HPP

#include "creatDataset.h"
#include "vl/hog.h"
#include "kv.hpp"
#include <memory>
#include <opencv2/ml/ml.hpp>
void svm_numCharRecognition(){
	std::vector<std::string> folder = { "0", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile = "charRecognition.xml";
	std::string a = "../resources/train/ann/charTrain/";
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::CHI2);
	svm->setDegree(14.0);//0.1
	// 1.4 bug fix: old 1.4 ver gamma is 1
	svm->setGamma(0.1);//0.1
	svm->setCoef0(1.0);//0.1
	svm->setC(11.39);//1
	svm->setNu(0.5);//0.1
	svm->setP(1.0);//0.1
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));
	//svm->setTermCriteria(cvTermCriteria(2, (int)1e7, 1e-6));
	std::vector<std::pair<std::string, int>> train_file_list;

	std::vector<std::string> train_list;

	for (auto i = 0; i < folder.size(); i++){
		train_list = utils::getFiles(a + folder[i]);
		std::cout << "The folder " << (a + folder[i]) << " has " << train_list.size() << " sample book" << std::endl;
		for (auto file : train_list)
			train_file_list.push_back(make_pair(file, i));
	}

	std::random_shuffle(train_file_list.begin(), train_file_list.end());
	std::cout << train_file_list.size() << std::endl;

	cv::Mat samples;
	std::vector<int> responses;
	vl_size numOrientations = 20;  //specifies the number of orientations
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[320];
	hogArray1 = (float*)vl_malloc(256 * sizeof(float));
	hogArray2 = (float*)vl_malloc(64 * sizeof(float));
	//extract hog 
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);
	for (auto f : train_file_list) {
		auto image = cv::imread(f.first, 0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
			continue;
		}
		cv::resize(image, image, Size(24, 24), 0, 0);
		float *vlimg = new float[576];
		int tmp = 0;
		for (int i = 0; i < 24; ++i){
			for (int j = 0; j < 24; ++j)
			{
				vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
			}
		}
		//set vl parameters
		vl_hog_set_use_bilinear_orientation_assignments(hog, true);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
		vl_hog_extract(hog, hogArray1);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
		vl_hog_extract(hog, hogArray2);
		for (int i = 0; i < 256; ++i)
			hogArray[i] = hogArray1[i];
		for (int i = 256; i < 320; ++i)
			hogArray[i] = hogArray2[i - 256];

		cv::Mat am(1, 320, CV_32FC1, hogArray);
		samples.push_back(am);
		responses.push_back(f.second);
	}

	cv::FileStorage fsData("../resources/train/ann/charsData.xml", cv::FileStorage::WRITE);
	fsData << "Samples" << samples;
	fsData << "Response" << responses;
	fsData.release();
	vl_hog_delete(hog);
	/*cv::Mat samples;
	std::vector<int> responses;
	cv::FileStorage fsData("../resources/train/charsJudge/charsData.xml", cv::FileStorage::READ);
	fsData["Samples"] >> samples;
	fsData["Response"] >> responses;
	fsData.release();*/
	auto train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
		responses);
	fprintf(stdout, ">> Training SVM model, please wait...\n");

	//cv::ml::ParamGrid Cgrid(10, 640, 2);//(7, 200, 1.2);
	//cv::ml::ParamGrid Ggrid(0.3, 0.9, 1.2);//(0.03, 0.2, 1.3);
	//gridCSearch(samples, responses, Cgrid, Ggrid, xmlFile);
	//cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),
	//cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA)
	
	cv::ml::ParamGrid Cgrid(1, 33, 1.7);//0.01 1000 5
	cv::ml::ParamGrid Ggrid(0.01, 0.3, 1.9);
	cv::ml::ParamGrid grid(0, 0, 0);
	svm->trainAuto(train_data, 5, Ggrid ,Cgrid, grid, grid, grid, grid, true);
	//svm->train(train_data);

	fprintf(stdout, ">> Training done.");
	fprintf(stdout, ">> Saving model file...\n");
	svm->save(a + xmlFile);

	fprintf(stdout, ">> Your SVM Model was saved to %s\n", xmlFile.c_str());
	fprintf(stdout, ">> Testing...\n");

}
void svm_charRTest() {
	// 1.4 bug fix: old 1.4 ver there is no null judge
	// if (NULL == svm_)
	std::vector<std::string> folder = { "0", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile = "../resources/train/ann/charRecognition.xml";
	std::string a = "../resources/train/ann/charTest/";
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(xmlFile);
	std::vector<std::pair<std::string, int>> test_file_list;

	std::vector<int> each_char_num;
	for (auto i = 0; i < folder.size(); i++){
		std::vector<std::string> train_list = utils::getFiles(a + folder[i]);
		if (!train_list.empty())
			each_char_num.push_back(train_list.size());
		else{
			std::cout << "Wrong file in " << (a + folder[i]) << std::endl;
			return;
		}

		for (auto file : train_list)
			test_file_list.push_back(make_pair(file, i));
	}

	std::random_shuffle(test_file_list.begin(), test_file_list.end());
	int count_all = test_file_list.size();

	vl_size numOrientations = 20;  //specifies the number of orientations
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[320];
	hogArray1 = (float*)vl_malloc(256 * sizeof(float));
	hogArray2 = (float*)vl_malloc(64 * sizeof(float));
	//extract hog 
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);

	int predict;
	int count_accuracy = 0;
	cv::Mat testMat_num(33, 33, CV_32SC1, cv::Scalar(0));
	cv::Mat testMat_ratio(33, 33, CV_64FC1, cv::Scalar(0.0));
	std::string imageName;
	for (auto f : test_file_list) {
		auto image = cv::imread(f.first, 0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
			continue;
		}
		if (image.size() != cv::Size(24, 24))
			cv::resize(image, image, Size(24, 24), 0, 0);
		float *vlimg = new float[576];
		int tmp = 0;
		for (int i = 0; i < 24; ++i){
			for (int j = 0; j < 24; ++j)
			{
				vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
			}
		}
		//set vl parameters
		vl_hog_set_use_bilinear_orientation_assignments(hog, true);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
		vl_hog_extract(hog, hogArray1);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
		vl_hog_extract(hog, hogArray2);
		for (int i = 0; i < 256; ++i)
			hogArray[i] = hogArray1[i];
		for (int i = 256; i < 320; ++i)
			hogArray[i] = hogArray2[i - 256];

		cv::Mat feature = cv::Mat(1, 320, CV_32FC1, hogArray);
		//std::cout << "feature: " << feature << std::endl;
		predict = svm->predict(feature);
		//std::cout << "predict: " << predict << std::endl;
		auto real = f.second;
		if (predict == real) count_accuracy++;
		testMat_num.at<int>(real, predict) += 1;
	}
	vl_hog_delete(hog);
	svm.release();
	std::cout << "count_all: " << count_all << std::endl;
	std::cout << "count_accuracy: " << (1.0*count_accuracy) / count_all << std::endl;
	std::cout << "the testMat_num is:" << std::endl;
	std::cout << testMat_num << std::endl;
	std::FILE *fp;
	fopen_s(&fp, (a + "libFile2").c_str(), "w+");
	for (int i = 0; i < each_char_num.size(); i++){
		for (int j = 0; j < each_char_num.size(); j++){
			testMat_ratio.at<double>(i, j) = (float)testMat_num.at<int>(i, j) / each_char_num[i];
			fprintf(fp, "%lf ", testMat_ratio.at<double>(i, j));
			//fprintf(fp, "%d:%d  ", itF, data[itF]);
		}
		fputs("\n", fp);
	}
	fclose(fp);
	cv::FileStorage fsData(a + "testMatfile.xml", cv::FileStorage::WRITE);
	fsData << "each_char_num" << each_char_num;
	fsData << "testMat_num" << testMat_num;
	fsData << "testMat_ratio" << testMat_ratio;
	fsData.release();
	return;
}
void numCharRecognition(){
	std::vector<std::string> folder = { "0", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile = "charRecognition.xml";
	std::string a = "../resources/train/ann/charTrain/";
	
	//svm->setTermCriteria(cvTermCriteria(2, (int)1e7, 1e-6));
	std::vector<std::pair<std::string, int>> train_file_list;

	std::vector<std::string> train_list;

	for (auto i = 0; i < folder.size(); i++){
	train_list = utils::getFiles(a + folder[i]);
	std::random_shuffle(train_list.begin(), train_list.end());
	//std::cout << "The folder " << (a + folder[i]) << " has " << train_list.size() << " sample book" << std::endl;
	for (int j = 0; j < 300 && j < train_list.size();++j)
		train_file_list.push_back(make_pair(train_list[j], i));
	}
	std::random_shuffle(train_file_list.begin(), train_file_list.end());
	///std::random_shuffle(train_file_list.begin(), train_file_list.end());
	std::cout << train_file_list.size() << std::endl;

	cv::Mat samples;
	std::vector<int> responses;
	vl_size numOrientations = 9;  //specifies the number of orientations20
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[155];
	hogArray1 = (float*)vl_malloc(124 * sizeof(float));// 320 256 64
	hogArray2 = (float*)vl_malloc(31 * sizeof(float));
	//extract hog
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);
	for (auto f : train_file_list) {
	auto image = cv::imread(f.first, 0);
	if (!image.data) {
	fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
	continue;
	}
	cv::resize(image, image, Size(24, 24), 0, 0);
	float *vlimg = new float[576];
	int tmp = 0;
	for (int i = 0; i < 24; ++i){
	for (int j = 0; j < 24; ++j)
	{
	vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
	}
	}
	//set vl parameters
	vl_hog_set_use_bilinear_orientation_assignments(hog, true);
	vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
	vl_hog_extract(hog, hogArray1);
	vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
	vl_hog_extract(hog, hogArray2);
	for (int i = 0; i < 124; ++i)
	hogArray[i] = hogArray1[i];
	for (int i = 124; i < 155; ++i)
	hogArray[i] = hogArray2[i - 124];

	cv::Mat am(1, 155, CV_32FC1, hogArray);
	samples.push_back(am);
	responses.push_back(f.second);
	}

	cv::Mat train_classes =
		cv::Mat::zeros((int)responses.size(), folder.size(), CV_32F);

	for (int i = 0; i < train_classes.rows; ++i) {
		train_classes.at<float>(i, responses[i]) = 1.f;
	}

	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
		train_classes);
	fprintf(stdout, ">> Training ANN model, please wait...\n");
	vl_hog_delete(hog);

	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
	int N = samples.cols;
	int m = folder.size();
	int first_hidden_neurons = int(std::sqrt((m + 2) * N) + 2 * std::sqrt(N / (m + 2)));
	int second_hidden_neurons = int(m * std::sqrt(N / (m + 2)));
	fprintf(stdout, ">> Use two-layers neural networks,\n");
	fprintf(stdout, ">> First_hidden_neurons: %d \n", first_hidden_neurons);
	fprintf(stdout, ">> Second_hidden_neurons: %d \n", second_hidden_neurons);

	cv::Mat layers(1, 4, CV_32SC1);
	layers.at<int>(0) =N;
	layers.at<int>(1) = first_hidden_neurons;
	layers.at<int>(2) = second_hidden_neurons;
	layers.at<int>(3) = m;
	ann->setLayerSizes(layers);
	ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
	ann->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 30000, 0.0001));
	ann->setBackpropWeightScale(0.1);
	ann->setBackpropMomentumScale(0.1);
	
	ann->train(train_data);
	ann->save(a + "ann.xml");
	std::cout << "Your ANN Model was saved to " << a+"ann.xml" << std::endl;
	return;

}
void charRTest() {
	// 1.4 bug fix: old 1.4 ver there is no null judge
	// if (NULL == svm_)
	std::vector<std::string> folder = { "0", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile = "../resources/train/ann/charTrain/ann.xml";
	std::string a = "../resources/train/ann/charTest/";
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load<cv::ml::SVM>(xmlFile);
	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::load(xmlFile);
	std::vector<std::pair<std::string, int>> test_file_list;

	std::vector<int> each_char_num;
	for (auto i = 0; i < folder.size(); i++){
		std::vector<std::string> train_list = utils::getFiles(a + folder[i]);
		if (!train_list.empty())
			each_char_num.push_back(train_list.size());
		else{
			std::cout << "Wrong file in " << (a + folder[i]) << std::endl;
			return;
		}

		for (auto file : train_list)
			test_file_list.push_back(make_pair(file, i));
	}

	std::random_shuffle(test_file_list.begin(), test_file_list.end());
	int count_all = test_file_list.size();

	vl_size numOrientations = 9;  //specifies the number of orientations20
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[155];
	hogArray1 = (float*)vl_malloc(124 * sizeof(float));// 320 256 64
	hogArray2 = (float*)vl_malloc(31 * sizeof(float));
	//extract hog 
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);
	
	int count_accuracy = 0;
	cv::Mat testMat_num(33, 33, CV_32SC1, cv::Scalar(0));
	cv::Mat testMat_ratio(33, 33, CV_64FC1, cv::Scalar(0.0));
	std::string imageName;
	for (auto f : test_file_list) {
		auto image = cv::imread(f.first, 0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
			continue;
		}
		if (image.size() != cv::Size(24, 24))
			cv::resize(image, image, Size(24, 24), 0, 0);
		float *vlimg = new float[576];
		int tmp = 0;
		for (int i = 0; i < 24; ++i){
			for (int j = 0; j < 24; ++j)
			{
				vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
			}
		}
		//set vl parameters
		vl_hog_set_use_bilinear_orientation_assignments(hog, true);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
		vl_hog_extract(hog, hogArray1);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
		vl_hog_extract(hog, hogArray2);
		for (int i = 0; i < 124; ++i)
			hogArray[i] = hogArray1[i];
		for (int i = 124; i < 155; ++i)
			hogArray[i] = hogArray2[i - 124];

		cv::Mat feature = cv::Mat(1, 155, CV_32FC1, hogArray);
		//std::cout << "feature: " << feature << std::endl;
		cv::Mat predictMat = cv::Mat::zeros(1, folder.size(), CV_32F);
		int predict = 0;
		 ann->predict(feature,predictMat);
		//std::cout << predictMat << std::endl;
		float maxScore = predictMat.at<float>(0, 0);
		for (int i = 1; i < predictMat.cols; ++i){
			if (predictMat.at<float>(0, i)>maxScore){
				maxScore = predictMat.at <float>(0,i);
				predict = i;
			}
		}
		//std::cout << "predict: " << predict << std::endl;
		auto real = f.second;
		if (predict == real) count_accuracy++;
		testMat_num.at<int>(real, predict) += 1;
	}
	vl_hog_delete(hog);
	ann.release();
	std::cout << "count_all: " << count_all << std::endl;
	std::cout << "count_accuracy: " << (1.0*count_accuracy) / count_all << std::endl;
	std::cout << "the testMat_num is:" << std::endl;
	std::cout << testMat_num << std::endl;
	std::FILE *fp;
	fopen_s(&fp, (a + "libFile2").c_str(), "w+");
	for (int i = 0; i < each_char_num.size(); i++){
		for (int j = 0; j < each_char_num.size(); j++){
			testMat_ratio.at<double>(i, j) = (float)testMat_num.at<int>(i, j) / each_char_num[i];
			fprintf(fp, "%lf ", testMat_ratio.at<double>(i, j));
			//fprintf(fp, "%d:%d  ", itF, data[itF]);
		}
		fputs("\n", fp);
	}
	fclose(fp);
	cv::FileStorage fsData(a + "testMatfile.xml", cv::FileStorage::WRITE);
	fsData << "each_char_num" << each_char_num;
	fsData << "testMat_num" << testMat_num;
	fsData << "testMat_ratio" << testMat_ratio;
	fsData.release();
	return;
}
void create_CSet(){
	std::vector<std::string> folder = { "zh_cuan", "zh_gan1", "zh_hei", "zh_jin", "zh_liao", "zh_min", "zh_qiong", "zh_sx", "zh_xin", "zh_yue", "zh_zhe", "zh_e", "zh_gui", "zh_hu", "zh_jing", "zh_lu", "zh_ning", "zh_shan", "zh_wan", "zh_yu", "zh_yun", "zh_gan", "zh_gui1", "zh_ji", "zh_jl", "zh_meng", "zh_qing", "zh_su", "zh_xiang", "zh_yu1", "zh_zang" };
	std::string aa1 = "../resources/train/chinese/annCh/";
	std::string aa2 = "../resources/train/chinese/trainChS/";
	std::vector<std::pair<std::string, int>> train_file_list;
	std::string aa3 = "../resources/train/chinese/train/";
	std::string aa4 = "../resources/train/chinese/test/";
	std::vector<std::string> train_list;
	std::shared_ptr<Kv> kv = std::shared_ptr<Kv>(new Kv);
	kv->load("../resources/train/chinese/province_mapping");
	for (auto i = 0; i < folder.size(); i++){
		train_list = utils::getFiles(aa2 + folder[i]);
		std::random_shuffle(train_list.begin(), train_list.end());
		int j = 0;
		std::string imageName;
		char file[100];
		for (; j < 100; ++j){
		cv::Mat imagenow = cv::imread(train_list[j], 0);
		imageName = Utils::getFileName(train_list[j], 1);
		cv::imwrite(aa4 + folder[i] +"/" + imageName, imagenow);

		}for (; j < train_list.size(); ++j){
		cv::Mat imagenow = cv::imread(train_list[j], 0);
		imageName = Utils::getFileName(train_list[j], 1);
		cv::imwrite(aa3 + folder[i] + "/" + imageName, imagenow);
		}
		/*std::string province = kv->get(folder[i]);
		std::cout << "The folder " << (aa1 + folder[i]) << province << " has " << train_list.size() << " sample book" << std::endl;
		for (auto file : train_list){
			train_file_list.push_back(make_pair(file, i));
		}
		if (train_list.size() < 10000){
			int count = 1;
			int a2 = 800 / train_list.size();
			int b2 = 800 % train_list.size();
			for (int j = 0; j < train_list.size(); ++j){
				cv::Mat imagenow = cv::imread(train_list[j], 0);
				
				cv::adaptiveThreshold(imagenow, imagenow,  255, ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,7,0);//5 -2
				//cv::Size char_size = cv::Size(44, 44);
				float ratio = (float)imagenow.cols / (float)imagenow.rows;
				if (ratio < 1){
					cv::resize(imagenow, imagenow, cv::Size(int(24 * ratio), 24), 0, CV_INTER_AREA);
					int expendW = (24 - imagenow.cols) / 2;
					cv::Mat out(24, 24, CV_8UC1, cv::Scalar(0));
					cv::Mat outRoi = out(cv::Rect(expendW, 0, imagenow.cols, imagenow.rows));
					imagenow.copyTo(outRoi);
					out.copyTo(imagenow);
				}
				else{
					cv::resize(imagenow, imagenow, cv::Size(24,int( 24/ratio)), 0, CV_INTER_AREA);
					int expendW = (24 - imagenow.rows) / 2;
					cv::Mat out(24, 24, CV_8UC1, cv::Scalar(0));
					cv::Mat outRoi = out(cv::Rect(0, expendW, imagenow.cols, imagenow.rows));
					imagenow.copyTo(outRoi);
					out.copyTo(imagenow);
				}
				
				char file[100];
				sprintf_s(file, "/%d.jpg", count);
				++count;
				cv::imwrite(aa2 + folder[i] + file, imagenow);
				for (int k = 0; k < a2; ++k){
					if ((j >= b2) && (k == a2 - 1))
						continue;
					int rand_type = rand();
					cv::Mat imageC;
					int wayT = rand_type % 4;
					switch (wayT)
					{
					case 0:{
							   int notation1 = (rand() % 2 == 0) ? -1 : 1;
							   int notation2 = (rand() % 2 == 0) ? -1 : 1;
							   int offsetx = notation1*max(rand() % 3, rand() % 3);
							   int offsety = notation2*max(rand() % 3, rand() % 3);
							   double a[6] = { 1, 0, offsetx, 0, 1, offsety };
							   cv::Mat trans_mat(2, 3, CV_64FC1, a);
							   warpAffine(imagenow, imageC, trans_mat, imagenow.size());
							   break;
					}
					case 1:{
							   int notation = (rand() % 2 == 0) ? -1 : 1;
							   float angle = notation*max(float(rand() % 7), float(rand() % 7));
							   cv::Point2f center(imagenow.cols / 2.0F, imagenow.rows / 2.0F);
							   Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);
							   warpAffine(imagenow, imageC, rot_mat, imagenow.size());
							   break;
					}

					case 2:{
							   int notation = (rand() % 2 == 0) ? -1 : 1;
							   float scale = 1 + notation*max((rand() % 7)*0.01, (rand() % 7)*0.01);
							   cv::Point2f center(imagenow.cols / 2.0F, imagenow.rows / 2.0F);
							   Mat rot_mat = getRotationMatrix2D(center, 0, scale);
							   int new_rows = round(imagenow.rows*scale);
							   int new_cols = round(imagenow.cols*scale);
							   int change_row = new_rows - imagenow.rows;
							   int change_col = new_cols - imagenow.cols;
							   warpAffine(imagenow, imageC, rot_mat, cv::Size(new_rows, new_cols));
							   if (change_row > 0){
								   imageC(cv::Rect(change_col / 2, change_row / 2, imagenow.cols, imagenow.rows)).copyTo(imageC);
							   }
							   else {
								   int boder1 = (-change_col % 2) == 0 ? 0 : 1;
								   int boder2 = (-change_row % 2) == 0 ? 0 : 1;
								   cv::copyMakeBorder(imageC, imageC, -change_row / 2, -change_row / 2 + boder2, -change_col, -change_col / 2 + boder1, BORDER_CONSTANT, cv::Scalar(0));
							   }
							   // std::cout << imageC.size() << std::endl;
							   break;

					}
					default:{
								Point2f dstTri[3];
								Point2f plTri[3];
								plTri[0] = Point2f(0, 0);
								plTri[1] = Point2f(imagenow.cols - 1, 0);
								plTri[2] = Point2f(0, imagenow.rows - 1);

								int a1 = rand() % 5 - 2;
								int a2 = rand() % 5 - 2;
								int a3 = rand() % 5 - 2;
								int a4 = rand() % 5 - 2;

								dstTri[0] = Point2f(a1, a4);
								dstTri[1] = Point2f(imagenow.cols - 1, a2);
								dstTri[2] = Point2f(a3, imagenow.rows - 1);
								Mat warp_mat = getAffineTransform(plTri, dstTri);
								warpAffine(imagenow, imageC, warp_mat, imagenow.size(), CV_INTER_CUBIC);
								break;

					}
					}
					cv::resize(imageC, imageC, cv::Size(24, 24), 0, 0);
					cv::threshold(imageC, imageC, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
					sprintf_s(file, "/%d.jpg", count);
					++count;
					cv::imwrite(aa2 + folder[i] + file, imageC);

				}
			}
		}*/
	}
}
void chinese_ann_train(){
	/*
	zh_cuan    ´¨
	zh_gan1    ¸Ê
	zh_hei     ºÚ
	zh_jin     ½ò
	zh_liao    ÁÉ
	zh_min     Ãö
	zh_qiong   Çí
	zh_sx      ½ú
	zh_xin     ÐÂ
	zh_yue     ÔÁ
	zh_zhe     Õã
	zh_e       ¶õ
	zh_gui     ¹ó
	zh_hu      »¦
	zh_jing    ¾©
	zh_lu      Â³
	zh_ning    Äþ
	zh_shan    ÉÂ
	zh_wan     Íî
	zh_yu      Ô¥
	zh_yun     ÔÆ
	zh_gan     ¸Ó
	zh_gui1    ¹ð
	zh_ji      ¼½
	zh_jl      ¼ª
	zh_meng    ÃÉ
	zh_qing    Çà
	zh_su      ËÕ
	zh_xiang   Ïæ
	zh_yu1     Óå
	zh_zang    ²Ø
	*/
	std::vector<std::string> folder = { "zh_cuan", "zh_gan1", "zh_hei", "zh_jin", "zh_liao", "zh_min", "zh_qiong", "zh_sx", "zh_xin", "zh_yue", "zh_zhe", "zh_e", "zh_gui", "zh_hu", "zh_jing", "zh_lu", "zh_ning", "zh_shan", "zh_wan", "zh_yu", "zh_yun", "zh_gan", "zh_gui1", "zh_ji", "zh_jl", "zh_meng", "zh_qing", "zh_su", "zh_xiang", "zh_yu1", "zh_zang" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile = "chinese_ann_Recognition.xml";
	std::string aa1 = "../resources/train/chinese/train/";

	std::string a = "../resources/train/chinese/";
	std::shared_ptr<Kv> kv = std::shared_ptr<Kv>(new Kv);
	kv->load("../resources/train/chinese/province_mapping");
	//svm->setTermCriteria(cvTermCriteria(2, (int)1e7, 1e-6));
	std::vector<std::pair<std::string, int>> train_file_list;

	std::vector<std::string> train_list;

	for (auto i = 0; i < folder.size(); i++){
		train_list = utils::getFiles(aa1 + folder[i]);
		std::string province = kv->get(folder[i]);
		std::cout << "The folder " << (aa1 + folder[i]) << province << " has " << train_list.size() << " sample book" << std::endl;
		std::random_shuffle(train_list.begin(), train_list.end());
		for (int j = 0; j < train_list.size(); ++j){//500

			train_file_list.push_back(make_pair(train_list[j], i));
		}
	}
	std::random_shuffle(train_file_list.begin(), train_file_list.end());
	///std::random_shuffle(train_file_list.begin(), train_file_list.end());
	std::cout << "The total number of all train book is: " << train_file_list.size() << std::endl;
	/*std::FILE *fptrain;
	fopen_s(&fptrain, (a + "caffe_data/train.txt").c_str(), "a+");
	//std::FILE *fptrain2;
	//fopen_s(&fptrain2, (a + "caffe_data/test.txt").c_str(), "a+");
	char filetrain[100];
	for (int count = 0; count < train_file_list.size(); ++count){
		auto image = cv::imread(train_file_list[count].first, 0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", train_file_list[count].first.c_str());
			continue;
		}
		cv::Mat img_bin;
		cv::copyMakeBorder(image, img_bin, 4, 4,4,4, BORDER_CONSTANT, Scalar(0));
		sprintf_s(filetrain, "%d.jpg", count+1);
		cv::imwrite(a + "caffe_data/train/" + filetrain, img_bin);
		cv::medianBlur(img_bin, img_bin, 3);
		fprintf(fptrain,"%s %d\n", filetrain, train_file_list[count].second);
		//fprintf(fptrain2, "%s\n", filetrain);
	}
	fclose(fptrain);
	//fclose(fptrain2);*/
	cv::Mat samples;
	std::vector<int> responses;
	vl_size numOrientations = 9;  //specifies the number of orientations20
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[155];
	hogArray1 = (float*)vl_malloc(124 * sizeof(float));// 320 256 64
	hogArray2 = (float*)vl_malloc(31 * sizeof(float));
	//extract hog
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);
	for (auto f : train_file_list) {
		auto image = cv::imread(f.first, 0);
		if (!image.data) {
			fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
			continue;
		}
		cv::resize(image, image, Size(24, 24), 0, 0);
		float *vlimg = new float[576];
		int tmp = 0;
		for (int i = 0; i < 24; ++i){
			for (int j = 0; j < 24; ++j)
			{
				vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
			}
		}
		//set vl parameters
		vl_hog_set_use_bilinear_orientation_assignments(hog, true);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
		vl_hog_extract(hog, hogArray1);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
		vl_hog_extract(hog, hogArray2);
		for (int i = 0; i < 124; ++i)
			hogArray[i] = hogArray1[i];
		for (int i = 124; i < 155; ++i)
			hogArray[i] = hogArray2[i - 124];

		cv::Mat am(1, 155, CV_32FC1, hogArray);
		samples.push_back(am);
		responses.push_back(f.second);
	}

	cv::Mat train_classes =
		cv::Mat::zeros((int)responses.size(), folder.size(), CV_32F);

	for (int i = 0; i < train_classes.rows; ++i) {
		train_classes.at<float>(i, responses[i]) = 1.f;
	}

	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
		train_classes);
	fprintf(stdout, ">> Training Chinese ANN model, please wait...\n");
	vl_hog_delete(hog);

	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
	int N = samples.cols;
	int m = folder.size();
	int first_hidden_neurons = int(1.5*std::sqrt((m + 2) * N) + 2 * std::sqrt(N / (m + 2)));
	int second_hidden_neurons = int(1.3*m * std::sqrt(N / (m + 2)));
	int third_hidden_neurons = int(0.9*m * std::sqrt(N / (m + 2)));
	fprintf(stdout, ">> Use two-layers neural networks,\n");
	fprintf(stdout, ">> First_hidden_neurons: %d \n", first_hidden_neurons);
	fprintf(stdout, ">> Second_hidden_neurons: %d \n", second_hidden_neurons);
	fprintf(stdout, ">> Third_hidden_neurons: %d \n", third_hidden_neurons);
	std::vector<double> momentN = { 0.3 };// , 0.5, 0.6, 0.65, 0.7, 0.9};
	std::vector<int> iterateN = { 70000 };// { 5, 50, 100, 200, 400, 800, 1600, 3200 };
	std::vector<double> learningN = { 0.0003 };//, 0.01, 0.1 };// { 0.0003, 0.001, 0.003, 0.01, 0.1 };
	cv::Mat layers(1, 5, CV_32SC1);
	layers.at<int>(0) = N;
	layers.at<int>(1) = first_hidden_neurons;
	layers.at<int>(2) = second_hidden_neurons;
	layers.at<int>(3) = third_hidden_neurons;
	layers.at<int>(4) = m;
	ann->setLayerSizes(layers);
	ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
	std::string a2 = "../resources/train/chinese/model/";
	char file[100];
	for (int i = 0; i < iterateN.size(); ++i){
		for (int j = 0; j < learningN.size(); ++j){
			for (int k = 0; k < momentN.size(); ++k){
				ann->setTermCriteria(cvTermCriteria(CV_TERMCRIT_NUMBER, iterateN[i], 0.0001));//30000 0.0001//300
				ann->setBackpropWeightScale(learningN[j]);//0.01
				ann->setBackpropMomentumScale(momentN[k]);//0.5
				sprintf_s(file, "%d %lf %lf.xml", iterateN[i], learningN[j],momentN[k]);
				ann->train(train_data);
				ann->save(a2 + file);
				std::cout << "Your ANN Model was saved to " << a2 +file << std::endl;
			}
		}
	}

	return;

}
void chinese_ann_charRTest() {
	// 1.4 bug fix: old 1.4 ver there is no null judge
	// if (NULL == svm_)
	std::vector<std::string> folder = { "zh_cuan", "zh_gan1", "zh_hei", "zh_jin", "zh_liao", "zh_min", "zh_qiong", "zh_sx", "zh_xin", "zh_yue", "zh_zhe", "zh_e", "zh_gui", "zh_hu", "zh_jing", "zh_lu", "zh_ning", "zh_shan", "zh_wan", "zh_yu", "zh_yun", "zh_gan", "zh_gui1", "zh_ji", "zh_jl", "zh_meng", "zh_qing", "zh_su", "zh_xiang", "zh_yu1", "zh_zang" };
	//std::string dst_path = ("../resources/train/charsJudge/has/");
	std::string xmlFile; //= "../resources/train/chinese/chinese_ann.xml";
	std::string aa2 = "../resources/train/chinese/test/";
	std::string a = "../resources/train/chinese/";
	//cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load<cv::ml::SVM>(xmlFile);
	/*std::vector<double> momentN ={ 0.3, 0.5, 0.6,0.65, 0.7, 0.9 };
	std::vector<int> iterateN = { 5, 50, 100, 200, 400, 800, 1600,3200 };
	std::vector<double> learningN =  {0.0003, 0.001,0.003, 0.01, 0.1};*/
	std::vector<double> momentN = { 0.3 };// , 0.5, 0.6, 0.65, 0.7, 0.9};
	std::vector<int> iterateN = { 70000 };// { 5, 50, 100, 200, 400, 800, 1600, 3200 };
	std::vector<double> learningN = { 0.0003 };//, 0.01, 0.1 };// { 0.0003, 0.001, 0.003, 0.01, 0.1 };
	std::string a2 = "../resources/train/chinese/model/";
	char file[100];
	for (int i = 0; i < iterateN.size(); ++i){
		for (int k = 0; k < momentN.size(); ++k){
			for (int j = 0; j < learningN.size(); ++j){
				sprintf_s(file, "%d %lf %lf.xml", iterateN[i], learningN[j], momentN[k]);
				xmlFile = a2+file;
				cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::load(xmlFile);
				std::vector<std::pair<std::string, int>> test_file_list;

				std::vector<int> each_char_num;
				for (auto mm = 0; mm < folder.size(); mm++){
					std::vector<std::string> train_list = utils::getFiles(aa2 + folder[mm]);
					if (!train_list.empty())
						each_char_num.push_back(train_list.size());
					else{
						std::cout << "Wrong file in " << (aa2 + folder[mm]) << std::endl;
						return;
					}

					for (auto file : train_list)
						test_file_list.push_back(make_pair(file, mm));
				}

				std::random_shuffle(test_file_list.begin(), test_file_list.end());
				int count_all = test_file_list.size();
				std::cout << "The ann model is (iterateNumber learningRatio momentum)  " << xmlFile << std::endl;
				vl_size numOrientations = 9;  //specifies the number of orientations20
				vl_size numChannels = 1;      //number of image channel
				vl_size height = 24;
				vl_size width = 24;
				vl_size cellSize = 12;     //size of a hog cell
				vl_size cellSize2 = 24;
				float *hogArray1;  //hog features array
				float *hogArray2;
				float  hogArray[155];
				hogArray1 = (float*)vl_malloc(124 * sizeof(float));// 320 256 64
				hogArray2 = (float*)vl_malloc(31 * sizeof(float));
				//extract hog 
				VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);
				int count_accuracy = 0;
				cv::Mat testMat_num(31, 31, CV_32SC1, cv::Scalar(0));
				cv::Mat testMat_ratio(31, 31, CV_64FC1, cv::Scalar(0.0));
				std::string imageName;
				for (auto f : test_file_list) {
					auto image = cv::imread(f.first, 0);
					if (!image.data) {
						fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
						continue;
					}
					if (image.size() != cv::Size(24, 24))
						cv::resize(image, image, Size(24, 24), 0, 0);
					float *vlimg = new float[576];
					int tmp = 0;
					for (int i = 0; i < 24; ++i){
						for (int j = 0; j < 24; ++j)
						{
							vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
						}
					}
					//set vl parameters
					vl_hog_set_use_bilinear_orientation_assignments(hog, true);
					vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
					vl_hog_extract(hog, hogArray1);
					vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
					vl_hog_extract(hog, hogArray2);
					for (int i = 0; i < 124; ++i)
						hogArray[i] = hogArray1[i];
					for (int i = 124; i < 155; ++i)
						hogArray[i] = hogArray2[i - 124];

					cv::Mat feature = cv::Mat(1, 155, CV_32FC1, hogArray);
					//std::cout << "feature: " << feature << std::endl;
					cv::Mat predictMat = cv::Mat::zeros(1, folder.size(), CV_32F);
					int predict = 0;
					ann->predict(feature, predictMat);
					//std::cout << predictMat << std::endl;
					float maxScore = predictMat.at<float>(0, 0);
					for (int i = 1; i < predictMat.cols; ++i){
						if (predictMat.at<float>(0, i)>maxScore){
							maxScore = predictMat.at <float>(0, i);
							predict = i;
						}
					}
					//std::cout << "predict: " << predict << std::endl;
					auto real = f.second;
					if (predict == real) count_accuracy++;
					testMat_num.at<int>(real, predict) += 1;
				}
				vl_hog_delete(hog);
				ann.release();
				//std::cout << "count_all: " << count_all << std::endl;
				std::cout << "count_accuracy: " << (1.0*count_accuracy) / count_all << std::endl;
				//std::cout << "the testMat_num is:" << std::endl;
				//std::cout << testMat_num << std::endl;
				std::FILE *fp;
				fopen_s(&fp, (a + "libFile").c_str(), "a+");
				fprintf(fp, "%d,   %lf,   %lf: %lf\n", iterateN[i], momentN[k], learningN[j], (1.0*count_accuracy) / count_all);
				for (int i = 0; i < each_char_num.size(); i++){
					for (int j = 0; j < each_char_num.size(); j++){
						testMat_ratio.at<double>(i, j) = (float)testMat_num.at<int>(i, j) / each_char_num[i];
						fprintf(fp, "%lf ", testMat_ratio.at<double>(i, j));
						//fprintf(fp, "%d:%d  ", itF, data[itF]);
					}
					fputs("\n", fp);
				}
				fclose(fp);
				cv::FileStorage fsData(a+ "testMatfile.xml", cv::FileStorage::APPEND);
				//fsData << "each_char_num" << each_char_num;
				//fsData << "testMat_num" << testMat_num;
				fsData << "count_accuracy" << (1.0*count_accuracy) / count_all;
				fsData << "testMat_ratio" << testMat_ratio;
				fsData.release();
			}
		}
	}
	
	return;
}
void num_alpha_recog(const CPlate& cplate,std::vector<string> &num_alpha) {
	std::string xmlFile = "../resources/model/ann/ann.xml";
	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::load(xmlFile);
	std::vector<std::pair<std::string, int>> test_file_list;

	vl_size numOrientations = 9;  //specifies the number of orientations20
	vl_size numChannels = 1;      //number of image channel
	vl_size height = 24;
	vl_size width = 24;
	vl_size cellSize = 12;     //size of a hog cell
	vl_size cellSize2 = 24;
	float *hogArray1;  //hog features array
	float *hogArray2;
	float  hogArray[155];
	hogArray1 = (float*)vl_malloc(124 * sizeof(float));// 320 256 64
	hogArray2 = (float*)vl_malloc(31 * sizeof(float));
	//extract hog 
	VlHog *hog = vl_hog_new(VlHogVariantUoctti, numOrientations, VL_FALSE);\

	std::shared_ptr<Kv> kv = std::shared_ptr<Kv>(new Kv);
	kv->load("../resources/model/ann/num_alpha.txt");
	for(int i=1;i<cplate.charMats.size();++i){
		cv::Mat image = cplate.charMats[i];
		bool is1 = cplate.is1[i];
		if (is1) {
			num_alpha.push_back(to_string(1));
			continue;
		}
		if (image.size() != cv::Size(24, 24))
			cv::resize(image, image, Size(24, 24), 0, 0);
		float *vlimg = new float[576];
		int tmp = 0;
		for (int i = 0; i < 24; ++i) {
			for (int j = 0; j < 24; ++j)
			{
				vlimg[tmp++] = image.at<uchar>(j, i) / 255.0;
			}
		}
		//set vl parameters
		vl_hog_set_use_bilinear_orientation_assignments(hog, true);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize);
		vl_hog_extract(hog, hogArray1);
		vl_hog_put_image(hog, vlimg, height, width, numChannels, cellSize2);
		vl_hog_extract(hog, hogArray2);
		for (int i = 0; i < 124; ++i)
			hogArray[i] = hogArray1[i];
		for (int i = 124; i < 155; ++i)
			hogArray[i] = hogArray2[i - 124];

		cv::Mat feature = cv::Mat(1, 155, CV_32FC1, hogArray);
		//std::cout << "feature: " << feature << std::endl;
		cv::Mat predictMat = cv::Mat::zeros(1, 33, CV_32F);
		int predict = 0;
		ann->predict(feature, predictMat);
		//std::cout << predictMat << std::endl;
		float maxScore = predictMat.at<float>(0, 0);
		for (int i = 1; i < predictMat.cols; ++i) {
			if (predictMat.at<float>(0, i)>maxScore) {
				maxScore = predictMat.at <float>(0, i);
				predict = i;
			}
		}
		num_alpha.push_back(kv->get(to_string(predict)));
	}
	vl_hog_delete(hog);
	ann.release();
	return;
}

#endif