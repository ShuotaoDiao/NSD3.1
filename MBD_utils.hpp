//
//  MBD_utils.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/30/22.
//

#ifndef MBD_utils_hpp
#define MBD_utils_hpp

#include <stdio.h>

#include "MBD_solver.hpp"
#include "NSD_utils.hpp"
#include "MBD_solver_batch.hpp"

std::vector<double> interface_mbd(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                   const std::vector<double>& x_init,
                  int sample_size,
                  double error);

void interface_mbd2(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                   const std::vector<double>& x_init,
                  int sample_init,
                  int max_it,
                  double error);

void interface_ombd(const std::string& folder_path,
                    const std::string& validation_folder_path,
                    const std::vector<double>& observed_predictor,
                    const std::vector<double>& x_init,
                    int sample_init,
                    int max_it,
                    double error);

void interface_ombd2(const std::string& folder_path,
                     const std::string& validation_folder_path,
                     const std::vector<double>& observed_predictor,
                     const std::vector<double>& x_init,
                     int sample_init,
                     int max_it,
                     const std::vector<int>& it_pointer,
                     double error);

// OMBD with incremental batch sampling
void interface_ombd2_batch(const std::string& folder_path,
                           const std::string& validation_folder_path,
                           const std::vector<double>& observed_predictor,
                           const std::vector<double>& x_init,
                           int batch_init,
                           int batch_incre,
                           int max_it,
                           const std::vector<int>& it_pointer,
                           double error);
#endif /* MBD_utils_hpp */
