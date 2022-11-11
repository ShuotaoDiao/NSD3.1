//
//  NSD_ioDB.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef NSD_ioDB_hpp
#define NSD_ioDB_hpp

#include <stdio.h>
#include <fstream>
#include <sstream>

#include "NSD_dataStructure.hpp"

std::vector<std::vector<dataPoint>> readNonparametricDB(std::string readPath); // read database from a text file

void printNonparametricDB(const std::vector<std::vector<dataPoint>>& dataPointDB);
// print dataPoint
void printDataPoint(const dataPoint& dataPoint01);
void inputDBTest(); // test on input functions of nonparametric DB

#endif /* NSD_ioDB_hpp */
