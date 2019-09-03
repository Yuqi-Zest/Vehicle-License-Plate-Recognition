#ifndef CNN_HPP
#define CNN_HPP
#include <opencv2/dnn.hpp>  
#include <opencv2/imgproc.hpp>  
#include <opencv2/highgui.hpp>  
#include <stdlib.h> 
#include "kv.hpp"
using namespace cv;
using namespace cv::dnn;

#include <fstream>  
#include <iostream>  
#include <cstdlib>  

/* Find best class for the blob (i. e. class with maximal probability) */
void getMaxClass(dnn::Blob &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix  
	Point classNumber;

	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

std::vector<std::string> readClassNames(const char *filename = "..//resources//model//cnn//synset_world.txt")//"D:\\opencv_contrib-master\\modules\\dnn\\samples\\synset_words.txt")
{
	std::vector<std::string> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}

	fp.close();
	return classNames;
}

void cnn_chinesesR(const cv::Mat &img,std::string &province)
{
	if (img.type() != CV_8UC1)
		return;
	if (img.size() == cv::Size(24, 24))
		copyMakeBorder(img, img, 4, 4, 4, 4, BORDER_CONSTANT, Scalar(0));
	else
	resize(img, img, Size(32, 32));       //GoogLeNet accepts only 224x224 RGB-images  
	String modelTxt = "..//resources//model//cnn//mylenet.prototxt";
	String modelBin = "..//resources//model//cnn//lenet_iter_20000.caffemodel";
	Ptr<dnn::Importer> importer;
	importer = dnn::createCaffeImporter(modelTxt, modelBin);
	dnn::Net net;
	importer->populateNet(net);
	importer.release();                     //We don't need importer anymore  
	
	dnn::Blob inputBlob = dnn::Blob(img);   //Convert Mat to dnn::Blob image batch  

	net.setBlob(".data", inputBlob);        //set the network input  

	net.forward();                          //compute output  
	dnn::Blob prob = net.getBlob("prob");   //gather output of "prob" layer  
	int classId;
	double classProb;
	getMaxClass(prob, &classId, &classProb);//find the best class  
	std::vector<std::string> classNames = readClassNames();
	//std::cout << "Best class: #" << classId << " '" << classNames[classId] << "'" << std::endl;
	//std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
	std::string chinese_pin = classNames[classId];
	std::shared_ptr<Kv> kv = std::shared_ptr<Kv>(new Kv);
	kv->load("../resources/train/chinese/province_mapping");
	province = kv->get(chinese_pin);

	return;
} //main  
#endif