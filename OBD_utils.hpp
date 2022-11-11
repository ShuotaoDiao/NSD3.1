//
//  OBD_utils.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/1/22.
//

#ifndef OBD_utils_hpp
#define OBD_utils_hpp

#include <stdio.h>

#include "OBD_solver.hpp"
#include "NSD_utils.hpp"

void interface_obd(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                  int N_pre,
                  int it_max,
                  double error);

#endif /* OBD_utils_hpp */
