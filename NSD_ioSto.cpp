//
//  NSD_ioSto.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include "NSD_ioSto.hpp"

// input funtions for random vectors
// including Ce
secondStageRHSmap readStochasticMap(const std::string& stochasticPath) {
    secondStageRHSmap RHS; // randomness of be, bi, and Ce on the right hand side of constraints in the second stage problem
    const std::string nameBeginSto("<sto>");
    const std::string nameEndSto("</sto>");
    const std::string nameBeginParameter_be("<be>");
    const std::string nameEndParameter_be("</be>");
    const std::string nameBeginParameter_bi("<bi>");
    const std::string nameEndParameter_bi("</bi>");
    const std::string nameBeginParameter_Ce("<Ce>");
    const std::string nameEndParameter_Ce("</Ce>");
    const std::string nameBeginParameter_Ci("<Ci>");
    const std::string nameEndParameter_Ci("</Ci>");
    std::string readCondition("null"); // current condition of reading
    const char* stochasticPathConst = stochasticPath.c_str();
    std::ifstream readFile(stochasticPathConst);
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile,line1)) {
            std::stringstream ss1(line1);
            if (nameBeginSto.compare(line1) != 0 && nameEndSto.compare(line1) != 0) {
                if (nameBeginParameter_be.compare(line1) == 0) {
                    readCondition = "be"; // start reading be
                }
                else if (nameBeginParameter_bi.compare(line1) == 0) {
                    readCondition = "bi"; // start reading bi
                }
                else if (nameBeginParameter_Ce.compare(line1) == 0){
                    readCondition = "Ce"; // start reading Ce
                }
                else if (nameBeginParameter_Ci.compare(line1) == 0) {
                    readCondition = "Ci"; // start reading Ci
                }
                else if (nameEndParameter_be.compare(line1) == 0) {
                    readCondition = "null"; // end reading be
                }
                else if (nameEndParameter_bi.compare(line1) == 0) {
                    readCondition = "null"; // end reading bi
                }
                else if (nameEndParameter_Ce.compare(line1) == 0) {
                    readCondition = "null"; // end reading Ce
                }
                else if (nameEndParameter_Ci.compare(line1) == 0) {
                    readCondition = "null"; // end reading Ci
                }
                else {
                    if (readCondition.compare("be") == 0) { // continue reading be
                        int location = 0;
                        ss1 >> location;
                        RHS.be_map.push_back(location); // store the location of randomness
                    }
                    if (readCondition.compare("bi") == 0) { // continue reading bi
                        int location = 0;
                        ss1 >> location;
                        RHS.bi_map.push_back(location); // store the location of randomness
                    }
                    if (readCondition.compare("Ce") == 0) { // continue reading Ce
                        std::pair<int, int> location;
                        std::string line2;
                        int count = 0;
                        while (getline(ss1, line2, ',')) {
                            std::stringstream ss2(line2);
                            if (count == 0) {
                                ss2 >> location.first;
                            }
                            else if (count == 1) {
                                ss2 >> location.second;
                            }
                            count += 1;
                        }
                        RHS.Ce_map.push_back(location);
                    }
                    if (readCondition.compare("Ci") == 0) { // continue reading Ci
                        std::pair<int, int> location;
                        std::string line2;
                        int count = 0;
                        while (getline(ss1, line2, ',')) {
                            std::stringstream ss2(line2);
                            if (count == 0) {
                                ss2 >> location.first;
                            }
                            else if (count == 1) {
                                ss2 >> location.second;
                            }
                            count += 1;
                        }
                        RHS.Ci_map.push_back(location);
                    }
                }
            }
        }
    }
    readFile.close();
    return RHS;
}


// merge randomVector
secondStageRHSpoint merge_randomVector(const dataPoint& be_point, const dataPoint& bi_point, const dataPoint& Ce_point, const dataPoint& Ci_point) {
    secondStageRHSpoint RHS_point;
    if (be_point.response.size() > 0) {
        RHS_point.be = be_point.response;
    }
    if (bi_point.response.size() > 0) {
        RHS_point.bi = bi_point.response;
    }
    if (Ce_point.response.size() > 0) {
        RHS_point.Ce = Ce_point.response;
    }
    if (Ci_point.response.size() > 0) {
        RHS_point.Ci = Ci_point.response;
    }
    // get predictor
    if (be_point.response.size() > 0) {
        RHS_point.predictor = be_point.predictor;
    }
    else if (bi_point.response.size() > 0) {
        RHS_point.predictor = bi_point.predictor;
    }
    else if (Ce_point.response.size() > 0) {
        RHS_point.predictor = Ce_point.predictor;
    }
    else if (Ci_point.response.size() > 0) {
        RHS_point.predictor = Ci_point.predictor;
    }
    else {
        std::cout << "Warning:(Merge Radam Vector) All the datapoints are empty!\n";
    }
    return RHS_point;
}
