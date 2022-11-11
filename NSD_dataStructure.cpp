//
//  NSD_dataStructure.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include "NSD_dataStructure.hpp"

double operator*(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double res = 0;
    if (vec1.size() != vec2.size()) {
        std::cout << "Warning: vec1.size() = " << vec1.size() << std::endl;
        std::cout << "Warning: vec2.size() = " << vec2.size() << std::endl;
        throw std::invalid_argument("Vector sizes do not match.\n");
    }
    for (int index = 0; index < vec1.size(); ++index) {
        res += vec1[index] * vec2[index];
    }
    return res;
}

std::vector<double> operator*(double a, const std::vector<double>& vec1) {
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(a * vec1[index]);
    }
    return res;
}

std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vector sizes are not matched.\n");
    }
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(vec1[index] + vec2[index]);
    }
    return res;
}

double max(double a, double b) {
    if (b > a) {
        return b;
    }
    else {
        return a;
    }
}

double min(double a, double b) {
    if (a < b) {
        return a;
    }
    else {
        return b;
    }
}
