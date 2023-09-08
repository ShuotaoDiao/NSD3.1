//
//  MBD_solver_batch.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 4/1/23.
//

#ifndef MBD_solver_batch_hpp
#define MBD_solver_batch_hpp

#include <stdio.h>

#include "MBD_solver.hpp"

// output solution at it_pointer with batch sampling
bd_output ombd_solver2_batch(const std::string& folder_path,
                       const std::vector<double>& observed_predictor,
                       const std::vector<double>& x_init,
                       int batch_init, // initial batch size
                       int batch_incre, // increment on the batch, nonnegative integer
                       int it_max,
                       const std::vector<int>& it_pointer,
                       double error);
#endif /* MBD_solver_batch_hpp */
