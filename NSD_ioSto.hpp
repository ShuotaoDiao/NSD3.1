//
//  NSD_ioSto.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef NSD_ioSto_hpp
#define NSD_ioSto_hpp

#include <stdio.h>

#include "NSD_dataStructure.hpp"

// including be, bi, Ce and Ci
secondStageRHSmap readStochasticMap(const std::string& stochasticPath);

// merge randomVector
secondStageRHSpoint merge_randomVector(const dataPoint& be_point, const dataPoint& bi_point, const dataPoint& Ce_point, const dataPoint& Ci_point);

#endif /* NSD_ioSto_hpp */
