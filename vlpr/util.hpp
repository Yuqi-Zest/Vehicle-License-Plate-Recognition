#ifndef PR_UTIL_UTIL_H_
#define PR_UTIL_UTIL_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <list>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <windows.h>
#include <direct.h>
#include <io.h>
#define PATH_DELIMITER '\\'

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif


#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p) \
if ((p)) {              \
	delete (p);         \
	(p) = NULL;         \
}
#endif

class Utils {
public:
	static long getTimestamp() {
		return static_cast<long>(cv::getTickCount());
	}

	/*
	* Get file name from a given path
	* bool postfix: including the postfix
	*/
	static std::string getFileName(const std::string &path,
		const bool postfix = false);

	/*
	* Split the given string into segements by a delimiter
	*/
	static std::vector<std::string> splitString(const std::string &str,
		const char delimiter);

	/*
	* returns the smaller of the two numbers
	*/
	template<typename T>
	static T min(const T &v1, const T &v2) {
		return (v1 < v2) ? v1 : v2;
	}

	/*
	* Get files from a given folder
	* all: including all sub-folders
	*/
	static std::vector<std::string> getFiles(const std::string &folder,
		const bool all = true);

	/*
	* Print string lines to std::out from an array of const char*,
	* this function is used for displaying command tips.
	* lines: should be end with (const char*)NULL.
	*/
	static void print_str_lines(const char** lines) {
		int index = 0;
		while (lines[index++]) {
			std::cout << lines[index - 1] << std::endl;
		}//wang::index++后，lines[index]不变，index+1
	}

	/*
	* Print string lines using {"string1", "string2"},
	* this is a easier way benefit from C++11.
	*/
	static void print_str_lines(const std::initializer_list<const char*> &lines) {
		for (auto line : lines) {
			std::cout << line << std::endl;
		}//wang::遍历算法
	}

	/*
	* Read and print by line.
	*/
	static void print_file_lines(const std::string &file) {
		std::ifstream fs(file);
		if (fs.good()) {
			while (!fs.eof()) {
				std::string line;
				std::getline(fs, line);

				line = utf8_to_gbk(line.c_str());

				std::cout << line << std::endl;
			}
			fs.close();
		}
		else {
			std::cerr << "cannot open file: " << file << std::endl;
		}
	}

	template<class T>
	static unsigned int levenshtein_distance(const T &s1, const T &s2) {
		const size_t len1 = s1.size(), len2 = s2.size();
		std::vector<unsigned int> col(len2 + 1), prevCol(len2 + 1);

		for (unsigned int i = 0; i < prevCol.size(); i++) prevCol[i] = i;
		for (unsigned int i = 0; i < len1; i++) {
			col[0] = i + 1;
			for (unsigned int j = 0; j < len2; j++)
				col[j + 1] = easypr::Utils::min(
				easypr::Utils::min(prevCol[1 + j] + 1, col[j] + 1),
				prevCol[j] + (s1[i] == s2[j] ? 0 : 1));
			col.swap(prevCol);
		}
		return prevCol[len2];
	}

	/*
	* Create multi-level directories by given folder.
	*/
	static bool mkdir(const std::string folder);

	/*
	* Make sure the destination folder exists,
	* if not, create it, then call cv::imwrite.
	*/
	static bool imwrite(const std::string &file, const cv::Mat &image);

	static std::string utf8_to_gbk(const char* utf8);


private:
	/*
	* Get the last slash from a path, compatible with Windows and *unix.
	*/
	static std::size_t get_last_slash(const std::string &path);
};

typedef Utils utils;

std::string Utils::getFileName(const std::string &path,
	const bool postfix /* = false */) {
	if (!path.empty()) {
		size_t last_slash = utils::get_last_slash(path);
		size_t last_dot = path.find_last_of('.');

		if (last_dot < last_slash || last_dot == std::string::npos) {
			// not found the right dot of the postfix,
			// return the file name directly
			return path.substr(last_slash + 1);
		}
		else {
			// the path has a postfix
			if (postfix) {
				// return the file name including postfix
				return path.substr(last_slash + 1);
			}
			// without postfix
			return path.substr(last_slash + 1, last_dot - last_slash - 1);
		}
	}
	return "";
}

std::vector<std::string> Utils::splitString(const std::string &str,
	const char delimiter) {
	std::vector<std::string> splited;
	std::string s(str);
	size_t pos;

	while ((pos = s.find(delimiter)) != std::string::npos) {
		std::string sec = s.substr(0, pos);

		if (!sec.empty()) {
			splited.push_back(s.substr(0, pos));
		}

		s = s.substr(pos + 1);
	}

	splited.push_back(s);

	return splited;
}

std::vector<std::string> Utils::getFiles(const std::string &folder,
	const bool all /* = true */) {
	std::vector<std::string> files;
	std::list<std::string> subfolders;
	subfolders.push_back(folder);

	while (!subfolders.empty()) {
		std::string current_folder(subfolders.back());

		if (*(current_folder.end() - 1) != '/') {
			current_folder.append("/*");
		}
		else {
			current_folder.append("*");
		}

		subfolders.pop_back();

		struct _finddata_t file_info;
		auto file_handler = _findfirst(current_folder.c_str(), &file_info);

		while (file_handler != -1) {
			if (all &&
				(!strcmp(file_info.name, ".") || !strcmp(file_info.name, ".."))) {
				if (_findnext(file_handler, &file_info) != 0) break;
				continue;
			}

			if (file_info.attrib & _A_SUBDIR) {
				// it's a sub folder
				if (all) {
					// will search sub folder
					std::string folder(current_folder);
					folder.pop_back();
					folder.append(file_info.name);

					subfolders.push_back(folder.c_str());
				}
			}
			else {
				// it's a file
				std::string file_path;
				// current_folder.pop_back();
				file_path.assign(current_folder.c_str()).pop_back();
				file_path.append(file_info.name);

				files.push_back(file_path);
			}

			if (_findnext(file_handler, &file_info) != 0) break;
		}  // while
		_findclose(file_handler);
	}

	return files;
}

bool Utils::mkdir(const std::string folder) {
	std::string folder_builder;
	std::string sub;
	sub.reserve(folder.size());
	for (auto it = folder.begin(); it != folder.end(); ++it) {
		const char c = *it;
		sub.push_back(c);
		if (c == PATH_DELIMITER || it == folder.end() - 1) {
			folder_builder.append(sub);
			if (0 != ::_access(folder_builder.c_str(), 0)) {
				// this folder not exist
				if (0 != ::_mkdir(folder_builder.c_str()))	// create failed
					return false;
			}
			sub.clear();
		}
	}
	return true;
}

bool Utils::imwrite(const std::string &file, const cv::Mat &image) {
	auto folder = file.substr(0, utils::get_last_slash(file));
	Utils::mkdir(folder);
	return cv::imwrite(file, image);
}


std::string Utils::utf8_to_gbk(const char* utf8) {
	int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
	wchar_t* wszGBK = new wchar_t[len + 1];
	memset(wszGBK, 0, len * 2 + 2);
	MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wszGBK, len);
	len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
	char* szGBK = new char[len + 1];
	memset(szGBK, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
	std::string strTemp(szGBK);
	if (wszGBK)
		delete[] wszGBK;
	if (szGBK)
		delete[] szGBK;
	return strTemp;
}


std::size_t Utils::get_last_slash(const std::string &path) {
	size_t last_slash_1 = path.find_last_of("\\");
	size_t last_slash_2 = path.find_last_of("/");
	size_t last_slash;

	if (last_slash_1 != std::string::npos && last_slash_2 != std::string::npos) {
		// C:/path\\to/file.postfix
		last_slash = std::max(last_slash_1, last_slash_2);
	}
	else {
		// C:\\path\\to\\file.postfix
		// C:/path/to/file.postfix
		last_slash =
			(last_slash_1 == std::string::npos) ? last_slash_2 : last_slash_1;
	}

	return last_slash;
}


#endif  // VLPR_UTIL_UTIL_H_