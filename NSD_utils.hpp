//
//  NSD_utils.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef NSD_utils_hpp
#define NSD_utils_hpp

#include <stdio.h>
#include "NSD_solver.hpp"

validationResult twoStageLP_validation_outputResults(const std::string& folder_path, const std::vector<double>& x_candidate);

void interface_nsd(const std::string& folder_path,
                     const std::string& validation_folder_path,
                     int max_iterations,
                     double f_upperbound,
                     double f_lowerbound,
                     double sigma_upperbound,
                     double sigma_lowerbound,
                     const std::vector<double>& observed_predictor,
                     int N_pre,
                     bool flag_debug);


#endif /* NSD_utils_hpp */
