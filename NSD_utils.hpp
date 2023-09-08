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
#include "NSD_solver_batch.hpp"

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

void interface_nsd2(const std::string& folder_path,
                    const std::string& validation_folder_path,
                    int max_iterations,
                    double f_upperbound,
                    double f_lowerbound,
                    double sigma_upperbound,
                    double sigma_lowerbound,
                    const std::vector<double>& observed_predictor,
                    const std::vector<int>& it_pointer,
                    int N_pre,
                    bool flag_debug);

void interface_nsd_batch(const std::string& folder_path,
                         const std::string& validation_folder_path,
                         int max_iterations,
                         double f_upperbound,
                         double f_lowerbound,
                         double sigma_upperbound,
                         double sigma_lowerbound,
                         const std::vector<double>& observed_predictor,
                         const std::vector<int>& it_pointer, // output estimated solution at the desired iteration
                         int batch_init, // initial batch size
                         int batch_incre, // increment on the batch, nonnegative integer
                         bool flag_debug);

#endif /* NSD_utils_hpp */
