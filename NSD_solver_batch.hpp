//
//  NSD_solver_batch.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 3/25/23.
//

#ifndef NSD_solver_batch_hpp
#define NSD_solver_batch_hpp

#include <stdio.h>
#include <stdlib.h> // rand
#include <ctime>
#include <cmath>

#include "NSD_solver.hpp"
// This module is a batch sampling version of NSD_solver

solverOutput nsd_batch_solver(const std::string& folder_path,
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

#endif /* NSD_solver_batch_hpp */
