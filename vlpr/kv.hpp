#ifndef EASYPR_UTIL_KV_H_
#define EASYPR_UTIL_KV_H_

#include <map>
#include <string>
#include "util.hpp"

class Kv {
public:
	Kv();

	void load(const std::string &file);

	std::string get(const std::string &key);

	void add(const std::string &key, const std::string &value);

	void remove(const std::string &key);

	void clear();

private:
	std::map<std::string, std::string> data_;
};


Kv::Kv() { }

void Kv::load(const std::string &file) {
	this->clear();

	std::ifstream reader(file);
	while (!reader.eof()) {
		std::string line;
		std::getline(reader, line);
		if (line.empty()) continue;

		const auto parse = [](const std::string &str) {
			std::string tmp, key, value;
			for (size_t i = 0, len = str.length(); i < len; ++i) {
				const char ch = str[i];
				if (ch == ' ') {
					if (i > 0 && str[i - 1] != ' ' && key.empty()) {
						key = tmp;
						tmp.clear();
					}
				}
				else {
					tmp.push_back(ch);
				}
				if (i == len - 1) {
					value = tmp;
				}
			}
			return std::make_pair(key, value);
		};

		auto kv = parse(line);
		this->add(kv.first, kv.second);
	}
	reader.close();
}

std::string Kv::get(const std::string &key) {
	if (data_.find(key) == data_.end()) {
		std::cerr << "[Kv] cannot find " << key << std::endl;
		return "";
	}
	return data_.at(key);
}

void Kv::add(const std::string &key, const std::string &value) {
	if (data_.find(key) != data_.end()) {
		fprintf(stderr,
			"[Kv] find duplicate: %s = %s , ignore\n",
			key.c_str(),
			value.c_str());
	}
	else {
		std::string v(value);
		v = utils::utf8_to_gbk(value.c_str());
		data_[key] = v;
	}
}

void Kv::remove(const std::string &key) {
	if (data_.find(key) == data_.end()) {
		std::cerr << "[Kv] cannot find " << key << std::endl;
		return;
	}
	data_.erase(key);
}

void Kv::clear() {
	data_.clear();
}

#endif // EASYPR_UTIL_KV_H_