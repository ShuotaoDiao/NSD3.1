//
//  OBD_solver.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/1/22.
//

#ifndef OBD_solver_hpp
#define OBD_solver_hpp

#include <stdio.h>

#include "NSD_solver.hpp"

// master problem
bd_solution obd_master(standardTwoStageParameters& model_parameters,
                      const std::vector<std::vector<double>>& subproblem_constraints_alpha,
                      const std::vector<std::vector<std::vector<double>>>& subproblem_constraints_beta,
                       const std::vector<int>& kNNSet);

// check new cuts
bool obd_check_new_cut(const std::vector<double>& alpha_set, const std::vector<std::vector<double>>& beta_set, double alpha, const std::vector<double>& beta);


// Online Benders Decomposition Solver
bd_output obd_solver(const std::string& folder_path,
                     const std::vector<double>& observed_predictor,
                     int N_pre,
                     int it_max,
                     double error);

#endif /* OBD_solver_hpp */
