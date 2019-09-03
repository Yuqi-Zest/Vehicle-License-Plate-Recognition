#ifndef CHARSJUDGE_CPP
#define CHARSJUDGE_CPP
#include "chars_judge.hpp"
namespace charsJudge{
	cv::Mat ProjectedHistogram(cv::Mat img, int t) {
		int sz = (t == 1) ? img.rows : img.cols;
		cv::Mat mhist = cv::Mat::zeros(1, sz, CV_16S);

		for (int j = 0; j < sz; j++) {
			cv::Mat data = (t == 1) ? img.row(j) : img.col(j);
			mhist.at<int>(j) = countNonZero(data);
		}
		// Normalize histogram
		mhist.convertTo(mhist, CV_32F, 1.0f / sz, 0);
		return mhist;
	}
	cv::Mat getHistogram(cv::Mat in) {
		const int VERTICAL = 0;
		const int HORIZONTAL = 1;

		// Histogram features
		cv::Mat vhist = ProjectedHistogram(in, VERTICAL);
		cv::Mat hhist = ProjectedHistogram(in, HORIZONTAL);

		// Last 10 is the number of moments components
		int numCols = vhist.cols + hhist.cols;

		cv::Mat out = cv::Mat::zeros(1, numCols, CV_32F);

		int j = 0;
		for (int i = 0; i < vhist.cols; i++) {
			out.at<float>(j) = vhist.at<float>(i);
			j++;
		}
		for (int i = 0; i < hhist.cols; i++) {
			out.at<float>(j) = hhist.at<float>(i);
			j++;
		}

		return out;
	}

	void getHistogramFeatures(const cv::Mat& image, cv::Mat& features) {
		cv::Mat grayImage;
		cv::cvtColor(image, grayImage, CV_RGB2GRAY);

		//grayImage = histeq(grayImage);

		cv::Mat img_threshold;
		cv::threshold(grayImage, img_threshold, 0, 255,
			CV_THRESH_OTSU + CV_THRESH_BINARY);
		features = getHistogram(img_threshold);
	}

	void svmTrain(std::string tureFile, std::string falseFile, std::string xmlFile) {
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setKernel(cv::ml::SVM::RBF);
		svm->setDegree(0.1);
		// 1.4 bug fix: old 1.4 ver gamma is 1
		svm->setGamma(0.1);
		svm->setCoef0(0.1);
		svm->setC(1);
		svm->setNu(0.1);
		svm->setP(0.1);
		svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));

		std::vector<std::pair<std::string, int>>train_file_list;

		auto has_file_train_list = utils::getFiles(tureFile);
		std::random_shuffle(has_file_train_list.begin(), has_file_train_list.end());

		auto no_file_train_list = utils::getFiles(falseFile);
		std::random_shuffle(no_file_train_list.begin(), no_file_train_list.end());


		for (auto file : has_file_train_list)
			train_file_list.push_back(make_pair(file, 1));

		for (auto file : no_file_train_list)
			train_file_list.push_back(make_pair(file, 0));

		cv::Mat samples;
		std::vector<int> responses;

		for (auto f : train_file_list) {
			auto image = cv::imread(f.first);
			if (!image.data) {
				fprintf(stdout, ">> Invalid image: %s  ignore.\n", f.first.c_str());
				continue;
			}
			cv::Mat feature;
			getHistogramFeatures(image, feature);
			feature = feature.reshape(1, 1);

			samples.push_back(feature);
			responses.push_back(f.second);
		}
		auto train_data = cv::ml::TrainData::create(samples, cv::ml::SampleTypes::ROW_SAMPLE,
			responses);
		fprintf(stdout, ">> Training SVM model, please wait...\n");

		//svm_->trainAuto(train_data, 10, SVM::getDefaultGrid(SVM::C),
		//                SVM::getDefaultGrid(SVM::GAMMA), SVM::getDefaultGrid(SVM::P),
		//                SVM::getDefaultGrid(SVM::NU), SVM::getDefaultGrid(SVM::COEF),
		//                SVM::getDefaultGrid(SVM::DEGREE), true);
		svm->train(train_data);

		fprintf(stdout, ">> Training done.");
		fprintf(stdout, ">> Saving model file...\n");
		svm->save(xmlFile);

		fprintf(stdout, ">> Your SVM Model was saved to %s\n", xmlFile);
		fprintf(stdout, ">> Testing...\n");
	}
}
#endif
