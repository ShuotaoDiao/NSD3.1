//
//  NSD_solver.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef NSD_solver_hpp
#define NSD_solver_hpp

#include <stdio.h>
#include <stdlib.h> // rand
#include <ctime>
#include <cmath>

#include "NSD_dataStructure.hpp"
#include "NSD_ioDB.hpp"
#include "NSD_ioModel.hpp"
#include "NSD_ioSto.hpp"

// supplemental fucntions
bool if_vec_equal(const std::vector<double>& vec1, const std::vector<double>& vec2);

// presolve
std::vector<double> nsd_presolve(standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

solverOutput nsd_solver(const std::string& folder_path,
                int max_iterations,
                double f_upperbound,
                double f_lowerbound,
                double sigma_upperbound,
                double sigma_lowerbound,
                const std::vector<double>& observed_predictor,
                int N_pre,
                bool flag_debug);


#endif /* NSD_solver_hpp */
