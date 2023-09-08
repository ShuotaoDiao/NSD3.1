//
//  MBD_solver.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/30/22.
//

#ifndef MBD_solver_hpp
#define MBD_solver_hpp

#include <stdio.h>

#include "NSD_solver.hpp"

// master problem
bd_solution mbd_master(standardTwoStageParameters& model_parameters,
                      std::vector<bd_subproblem_cuts>& subproblem_constraints,
                      int k);

// Multi-cut Benders Decomposition Solver
bd_output mbd_solver(const std::string& folder_path,
                     const std::vector<double>& observed_predictor,
                     const std::vector<double>& x_init,
                     int sample_size,
                     double error);

// Online Multi-Cut Benders Decomposition Solver
bd_output ombd_solver(const std::string& folder_path,
                      const std::vector<double>& observed_predictor,
                      const std::vector<double>& x_init,
                      int sample_init,
                      int it_max,
                      double error);

// output solution at it_pointer
bd_output ombd_solver2(const std::string& folder_path,
                       const std::vector<double>& observed_predictor,
                       const std::vector<double>& x_init,
                       int sample_init,
                       int it_max,
                       const std::vector<int>& it_pointer,
                       double error);


#endif /* MBD_solver_hpp */
