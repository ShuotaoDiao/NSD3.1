//
//  MBD_solver_batch.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 4/1/23.
//

#include "MBD_solver_batch.hpp"

bd_output ombd_solver2_batch(const std::string& folder_path,
                       const std::vector<double>& observed_predictor,
                       const std::vector<double>& x_init,
                       int batch_init, // initial batch size
                       int batch_incre, // increment on the batch, nonnegative integer
                       int it_max,
                       const std::vector<int>& it_pointer,
                             double error) {
    // timer
    std::clock_t time_start;
    time_start = std::clock();
    // current time
    std::time_t currTime = std::time(nullptr);
    // STEP 1: INITIALIZATION
    // algorithm parameters
    double sigma = 1.0;
    //double beta = 0.5; 0.6 is okay
    double beta = 0.6; // 0 < beta < 1
    int k = (int) pow(batch_init, beta);
    int k_new = 0;
    double knn_radius = 0;
    std::vector<double> distanceSet;
    std::vector<int> orderSet;
    std::vector<int> kNNSet;
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
    bool flag_Ce; // tell if Ce stochastic is generated
    bool flag_Ci; // tell if Ci stochastic is generated
    std::vector<secondStageRHSpoint> RHS_dataset;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    std::string resultsOutput_path = folder_path + "/computationalResults(ombd_batch_v1.1).txt";
    std::string resultsSolution_path = folder_path + "/sol(ombd_batch_v1.1).txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    // create stream object
    std::ifstream readFile_be(be_DB_path_const);
    std::ifstream readFile_bi(bi_DB_path_const);
    std::ifstream readFile_Ce(Ce_DB_path_const);
    std::ifstream readFile_Ci(Ci_DB_path_const);
    // create database
    std::vector<std::vector<dataPoint>> be_DB;
    std::vector<std::vector<dataPoint>> bi_DB;
    std::vector<std::vector<dataPoint>> Ce_DB;
    std::vector<std::vector<dataPoint>> Ci_DB;
    // create model structure
    standardTwoStageParameters model_parameters;
    // create sto object
    secondStageRHSmap RHSmap;
    // read be
    if (readFile_be.is_open()) {
        std::cout << "be_DB is found." << std::endl;
        readFile_be.close(); // close the file
        // read be database
        be_DB = readNonparametricDB(be_DB_path);
        flag_be = true;
    }
    else {
        readFile_be.close(); // close the file
        flag_be = false;
        std::cout << "be_DB is not found!" << std::endl;
    }
    // read bi
    if (readFile_bi.is_open()) {
        std::cout << "bi_DB is found." << std::endl;
        readFile_be.close(); // close the file
        // read bi database
        bi_DB = readNonparametricDB(bi_DB_path);
        flag_bi = true;
    }
    else {
        readFile_bi.close(); // close the file
        flag_bi = false;
        std::cout << "bi_DB is not found!" << std::endl;
    }
    // read Ce
    if (readFile_Ce.is_open()) {
        std::cout << "Ce_DB stochastic part is found." << std::endl;
        readFile_Ce.close(); // close the file
        // Ce database
        Ce_DB = readNonparametricDB(Ce_DB_path);
        flag_Ce = true;
    }
    else {
        readFile_Ce.close(); // close the file
        flag_Ce = false;
        std::cout << "Ce_DB is not found!" << std::endl;
    }
    // read Ci
    if (readFile_Ci.is_open()) {
        std::cout << "Ci_DB stochastic part is found." << std::endl;
        readFile_Ci.close(); // close the file
        // Ci database
        Ci_DB = readNonparametricDB(Ci_DB_path);
        flag_Ci = true;
    }
    else {
        readFile_Ci.close(); // close the file
        flag_Ci = false;
        std::cout << "Ci_DB is not found!" << std::endl;
    }
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // estimated solutions
    std::vector<double> x_est(model_parameters.A.getColLength(), 0.0);
    // read sto file
    RHSmap = readStochasticMap(sto_path);
    // STEP 2: SOLVING PROCESS (SD-kNN)
    // initialize feasibility cut collection
    std::vector<feasibilityCut> feasibility_cuts;
    // initialization of output file
    const char* writeFilePath = resultsOutput_path.c_str();
    std::fstream writeFile;
    writeFile.open(writeFilePath,std::fstream::app); // append results to the end of the file
    //
    const char* writeFileSolPath = resultsSolution_path.c_str();
    std::fstream writeFileSol;
    writeFileSol.open(writeFileSolPath,std::fstream::app);
    // write initial setup
    std::cout << "*******************************************\n";
    writeFile << "*******************************************\n";
    std::cout << "Online Multi-Cut Benders Decomposition with Incremental Batch Sampling (v1.1) is initialized\n";
    writeFile << "Online Multi-Cut Benders Decomposition with Incremental Batch Sampling (v1.1) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, beta, Initial Batch Size, Increment on the Batch Size, error" << std::endl;
    writeFile << "sigma, beta, Initial Batch Size, Increment on the Batch Size, error" << std::endl;
    std::cout << sigma << ", " << beta << ", " << batch_init << ", " << batch_incre << ", " << error << std::endl;
    writeFile << sigma << ", " << beta << ", " << batch_init << ", " << batch_incre << ", " << error << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.getRowLength() << ", " << model_parameters.A.getColLength() << std::endl;
    writeFile << model_parameters.A.getRowLength() << ", " << model_parameters.A.getColLength() << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.getRowLength() << ", " << model_parameters.D.getColLength() << std::endl;
    writeFile << model_parameters.D.getRowLength() << ", " << model_parameters.D.getColLength() << std::endl;
    // set up initial incumbent solution
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int predictor_index = 0; predictor_index < observed_predictor.size() - 1; ++predictor_index) {
        std::cout << observed_predictor[predictor_index] << ", ";
        writeFile << observed_predictor[predictor_index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    // initial solution
    std::cout << "Initial Solution: ";
    writeFile << "Initial Solution: ";
    for (int x_idx = 0; x_idx < model_parameters.A.getColLength() - 1; ++x_idx) {
        std::cout << x_init[x_idx] << ", ";
        writeFile << x_init[x_idx] << ", ";
    }
    std::cout << x_init[model_parameters.A.getColLength() - 1] << std::endl;
    writeFile << x_init[model_parameters.A.getColLength() - 1] << std::endl;
    // set up environment for the subproblem
    long y_size = model_parameters.D.getColLength();
    // set up the model
    IloEnv env_sub;
    IloModel mod_sub(env_sub);
    IloNumVarArray y(env_sub,y_size,model_parameters.y_lb,IloInfinity,ILOFLOAT);
    mod_sub.add(y);
    // objective function
    IloExpr expr_obj_sub(env_sub);
    for (int idx = 0; idx < model_parameters.d.getNzeroLen(); ++idx) {
        expr_obj_sub += model_parameters.d.getVal(idx) * y[model_parameters.d.getLoc(idx)];
    }
    IloObjective obj_sub = IloMinimize(env_sub,expr_obj_sub);
    mod_sub.add(obj_sub);
    // stndard form equality constraints [D Islack] [y s] + Cx = e
    IloRangeArray constraintsEquality_sub(env_sub);
    std::vector<IloExpr> exprs_eq_sub;
    for (int idx = 0; idx < model_parameters.D.getRowLength(); ++idx) {
        IloExpr expr(env_sub);
        exprs_eq_sub.push_back(expr);
    }
    // coefficients before y; Dy
    for (int col_idx = 0; col_idx < model_parameters.D.getColLength(); ++col_idx) {
        int beg_idx = model_parameters.D.getCbeg(col_idx);
        for (int idx = beg_idx; idx < model_parameters.D.getClen(col_idx) + beg_idx; ++idx) {
            exprs_eq_sub[model_parameters.D.getRow(idx)] += model_parameters.D.getVal(idx) * y[col_idx];
        }
    }
    // coefficients before x; Cx
    for (int col_idx = 0; col_idx < model_parameters.C.getColLength(); ++col_idx) {
        int beg_idx = model_parameters.C.getCbeg(col_idx);
        for (int idx = beg_idx; idx < model_parameters.C.getClen(col_idx) + beg_idx; ++idx) {
            exprs_eq_sub[model_parameters.C.getRow(idx)] += model_parameters.C.getVal(idx) * x_est[col_idx];
        }
    }
    // right hand side e
    for (int idx = 0; idx < model_parameters.e.getNzeroLen(); ++idx) {
        exprs_eq_sub[model_parameters.e.getLoc(idx)] -= model_parameters.e.getVal(idx);
    }
    // add the equality constraints
    for (int idx = 0; idx< model_parameters.D.getRowLength(); ++idx) {
        constraintsEquality_sub.add(exprs_eq_sub[idx] == 0);
    }
    mod_sub.add(constraintsEquality_sub);
    // set up cplex solver for the subproblem
    IloCplex cplex_sub(env_sub);
    cplex_sub.extract(mod_sub);
    cplex_sub.setOut(env_sub.getNullStream());
    // intermediate values for rhs update
    // deterministic part of the rhs_bounds
    std::vector<double> e_det(model_parameters.D.getRowLength(), 0.0);
    for (int idx = 0; idx < model_parameters.e.getNzeroLen(); ++idx) {
        e_det[model_parameters.e.getLoc(idx)] += model_parameters.e.getVal(idx);
    }
    std::vector<double> det_rhs_bounds = e_det;
    std::vector<double> rhs_bounds(model_parameters.D.getRowLength(), 0.0);
    // end settig up subproblem ==================================
    // ===kNN Estimation Initialization===
    std::cout << "===kNN Estimation Initialization===\n";
    writeFile << "===kNN Estimation Initialization===\n";
    // find the kNN set
    for (int idx_pre = 0; idx_pre < batch_init; ++idx_pre) {
        k_new = (int) pow(batch_init, beta);
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_pre];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_pre];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_pre];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_pre];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        RHS_dataset.push_back(RHS_datapoint);
        // calculate the squared distance
        double distance_squared = 0;
        for (int idx = 0; idx < RHS_datapoint.predictor.size(); ++idx) {
            distance_squared += (RHS_datapoint.predictor[idx] - observed_predictor[idx]) * (RHS_datapoint.predictor[idx] - observed_predictor[idx]);
        }
        distanceSet.push_back(distance_squared);
        // store the new squared distance
        // sorting (like insert sorting)
        if (idx_pre == 0) { // first iteration
            orderSet.push_back(1);
        }
        else { // from left to right in increasing order
            //int left_index = 0; // the index corresponds to the largest distance that is smaller than the current one
            //double left_distance = -1;
            // double indices used for tie-breaking
            int right_index = -1; // the index corresponds to the smallest distance that is larger than  the current one
            double right_distance = -1;
            for (int index = 0; index < orderSet.size(); ++index) {
                /*
                if (distanceSet[index] < distance_squared) {
                    if (left_index == 0) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                    else if (distanceSet[index] > left_distance) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                }
                 */
                if (distanceSet[index] > distance_squared) {
                    if (right_index == -1) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                    else if (distanceSet[index] < right_distance) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                    else if (distanceSet[index] == right_distance && right_index > orderSet[index]) {
                        right_index = orderSet[index];
                    }
                }
            }
            /*
            if (flag_debug == true) {
                std::cout << "Output double indices\n";
                writeFile << "Output double indices\n";
                std::cout << "left index: " << left_index << std::endl;
                writeFile << "left index: " << left_index << std::endl;
                std::cout << "right index: " << right_index << std::endl;
                writeFile << "right index: " << right_index << std::endl;
            }
             */
            // update the orderSet
            for (int index = 0; index < orderSet.size(); ++index) {
                if (right_index != -1 && orderSet[index] >= right_index) {
                    orderSet[index] = orderSet[index] + 1;
                }
                //if (left_index == 0) { // current one is the nearest neighbor
                //    orderSet[index] = orderSet[index] + 1;
                //}
                //else if (orderSet[index] > left_index) {
                //    orderSet[index] = orderSet[index] + 1;
                //}
            }
            if (right_index == -1) {
                orderSet.push_back((int) orderSet.size() + 1);
            }
            else {
                orderSet.push_back(right_index);
            }
            /*
            if (flag_debug == true) {
                std::cout << "Updated Order in the scenario set\n";
                writeFile << "Updated Order in the scenario set\n";
                std::cout << "Index, Order, Distance (Squared)\n";
                writeFile << "Index, Order, Distance (Squared)\n";
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    std::cout << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    writeFile << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    if (orderSet[index] <= k_new) {
                        std::cout << "*";
                        writeFile << "*";
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
                    }
                }
            }
             */
        }
    }
    // update the kNN set
    kNNSet.clear(); // clear the old kNN set
    for (int index = 0; index < orderSet.size(); ++index) {
        if (orderSet[index] <= k_new) {
            kNNSet.push_back(index);
            if (orderSet[index] == k_new) {
                knn_radius = distanceSet[index];
            }
        }
    }
    // ===End kNN Initialization
    // STEP 3 MAIN LOOP
    std::cout << "Start Main Loop\n";
    writeFile << "Start Main Loop\n";
    int N = batch_init; // sample size
    int idx_datapoint = batch_init - 1; // index for the last data point
    int batch_cur = batch_init; // current batch size
    k = k_new;
    bd_output res;
    res.x = x_init;
    int it_count = 0;
    int idx_pointer = 0;
    // main loop for incremental batch sampling
    for (int it = 0; it < it_max; ++it) {
        std::cout << "****************************************\n";
        writeFile << "****************************************\n";
        std::cout << "Main Iteration: " << it << std::endl;
        writeFile << "Main Iteration: " << it << std::endl;
        // solve the PSP instance
        // MBD step
        // initialize z vector for storing the estimated costs of each second stage problem
        std::vector<double> z_est(k,0.0);
        // initialize constraints of the subproblem
        std::vector<bd_subproblem_cuts> subproblem_cons;
        for (int idx_subproblem = 0; idx_subproblem < k; ++idx_subproblem) {
            bd_subproblem_cuts sub_cuts;
            subproblem_cons.push_back(sub_cuts);
        }
        bool flag_init = true;
        bool flag_terminate = false;
        double max_gap = -1e10;
        int it_num = 1;
        double res_max_gap = 1e10;
        while (flag_terminate == false) {
            it_count++;
            std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            writeFile << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            std::cout << "(MBD) Iteration:  " << it_num << std::endl;
            writeFile << "(MBD) Iteration:  " << it_num << std::endl;
            // solve the relaxed master problem
            // update estimated solution (step 1 of the Benders decomposition)
            bd_solution est_sol = mbd_master(model_parameters, subproblem_cons, k);
            if (est_sol.sol_flag == 0) {
                std::cout << "Master problem is feasible.\n";
                writeFile << "Master problem is feasible.\n";
                x_est = est_sol.x;
                z_est = est_sol.z;
            }
            else if (it_num > 1) {
                std::cout << "Warning: Master problem is infeasible. The program is terminated.\n";
                writeFile << "Warning: Master problem is infeasible. The program is terminated.\n";
                break;
            }
            else {
                std::cout << "First iteration, Master problem is infeasible or unbounded.\n";
                writeFile << "First iteration, Master problem is infeasible or unbounded.\n";
                std::cout << "Initial Solution is used as the starting point.\n";
                writeFile << "Initial Solution is used as the starting point.\n";
                for (int idx_knn = 0; idx_knn < k; ++idx_knn) {
                    z_est[idx_knn] = -1e10;
                }
                for (int idx_x = 0; idx_x < model_parameters.A.getColLength(); ++idx_x) {
                    x_est[idx_x] = res.x[idx_x];
                }
            }
            flag_terminate = true;
            // update deterministic part of the rhs_bounds at current x_est
            // Dy = [e - Cx]
            det_rhs_bounds = e_det;
            // det C x
            for (int col_idx = 0; col_idx < model_parameters.C.getColLength(); ++col_idx) {
                int beg_idx = model_parameters.C.getCbeg(col_idx);
                for (int idx = beg_idx; idx < model_parameters.C.getClen(col_idx) + beg_idx; ++idx) {
                    det_rhs_bounds[model_parameters.C.getRow(idx)] -= model_parameters.C.getVal(idx) * x_est[col_idx];
                }
            }
            for (int knn_idx = 0; knn_idx < kNNSet.size(); ++knn_idx) {
                // update the subproblem
                for (int idx_row = 0; idx_row < model_parameters.D.getRowLength(); ++idx_row) {
                    rhs_bounds[idx_row] = det_rhs_bounds[idx_row];
                } // update the deterministic part
                // update the stochastic parts of e
                for (int idx_be = 0; idx_be < RHS_dataset[kNNSet[knn_idx]].be.size(); ++idx_be) {
                    rhs_bounds[RHSmap.be_map[idx_be]] += RHS_dataset[kNNSet[knn_idx]].be[idx_be];
                }
                // right hand side (stochastic part) equality bi_(i) inequality
                for (int idx_bi = 0; idx_bi < RHS_dataset[kNNSet[knn_idx]].bi.size(); ++idx_bi) {
                    rhs_bounds[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] += RHS_dataset[kNNSet[knn_idx]].bi[idx_bi];
                }
                // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[knn_idx]].Ce.size(); ++idx_Ce) {
                    rhs_bounds[RHSmap.Ce_map[idx_Ce].first] -= RHS_dataset[kNNSet[knn_idx]].Ce[idx_Ce] * x_est[RHSmap.Ce_map[idx_Ce].second];
                }
                // coefficients before x (stochastic part) inequality (location is behind equality constraints)
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[knn_idx]].Ci.size(); ++idx_Ci) {
                    rhs_bounds[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] -= RHS_dataset[kNNSet[knn_idx]].Ci[idx_Ci] * x_est[RHSmap.Ci_map[idx_Ci].second];
                }
                
                // update the RHS
                for (int idx_row = 0; idx_row < rhs_bounds.size(); ++idx_row) {
                    constraintsEquality_sub[idx_row].setBounds(rhs_bounds[idx_row], rhs_bounds[idx_row]);
                    // reset rhs_bounds to 0
                    rhs_bounds[idx_row] = 0;
                }
                // calculate the dual multipliers
                IloBool flag_solve = cplex_sub.solve();
                if (flag_solve == IloTrue) {
                    IloNumArray dual_equality_sub(env_sub);
                    std::vector<double> pi; // debug sign of pi !!
                    double optimal_value = cplex_sub.getObjValue(); // get the optimal value
                    cplex_sub.getDuals(dual_equality_sub,constraintsEquality_sub);
                    for (int index_eq = 0; index_eq < model_parameters.D.getRowLength(); ++index_eq) {
                        double pi_temp = dual_equality_sub[index_eq]; // move y to the right hand side
                        pi.push_back(pi_temp);
                    }
                    // compute the optimality cut
                    // deterministic e
                    double pi_e = model_parameters.e.fast_dotProduct(pi);
                    // stochastic e
                    // equality part
                    for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[knn_idx]].be.size(); ++idx_eq) {
                        pi_e += pi[idx_eq] * RHS_dataset[kNNSet[knn_idx]].be[idx_eq];
                    }
                    // inequality part (before standardizing) inequality constraint is after the equality constraints
                    for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[knn_idx]].bi.size(); ++idx_ineq) {
                        pi_e += pi[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[knn_idx]].bi[idx_ineq];
                    }
                    // slope
                    std::vector<double> neg_pi_C = model_parameters.C.fast_rightMultiply(pi);
                    // negate
                    for (int idx2 = 0; idx2 < neg_pi_C.size(); ++idx2) {
                        neg_pi_C[idx2] = (-1.0) * neg_pi_C[idx2];
                    }
                    // stochastic C
                    // equality
                    for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[knn_idx]].Ce.size(); ++idx_Ce) {
                        neg_pi_C[RHSmap.Ce_map[idx_Ce].second] -= RHS_dataset[kNNSet[knn_idx]].Ce[idx_Ce] * pi[RHSmap.Ce_map[idx_Ce].first];
                    }
                    // inequality before standardizing
                    for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[knn_idx]].Ci.size(); ++idx_Ci) {
                        neg_pi_C[RHSmap.Ci_map[idx_Ci].second] -= RHS_dataset[kNNSet[knn_idx]].Ci[idx_Ci] * pi[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                    }
                    // construct a new cut
                    subproblem_cons[knn_idx].alpha_array.push_back(pi_e);
                    subproblem_cons[knn_idx].beta_array.push_back(neg_pi_C);
                    if (est_sol.sol_flag == 0 && optimal_value > z_est[knn_idx] + error) {
                        flag_terminate = false;
                    }
                    double sub_gap = optimal_value - z_est[knn_idx];
                    //std::cout << "Infeasibility of the optimality cut (Subproblem " << idx_subproblem << "):" << sub_gap << std::endl;
                    if (max_gap < sub_gap) {
                        max_gap = sub_gap;
                    }
                } // end if (flag_solve == IloTrue)
                else {
                    throw std::logic_error("OBD solver: Main solving process terminate. Subproblem is infeasible.\n");
                }
            } // end for (int knn_idx = 0; knn_idx < kNNSet.size(); ++knn_idx)
            if (est_sol.sol_flag != 0 && flag_init == true) { // first iteration is infeasible or unbounded
                flag_terminate = false;
            }
            flag_init = false;
            std::cout << "Max Gap (2nd Stage Recourse): " << max_gap << std::endl;
            writeFile << "Max Gap (2nd Stage Recourse): " << max_gap << std::endl;
            std::cout << "Estimated First-Stage Solution: \n";
            writeFile << "Estimated First-Stage Solution: \n";
            for (int idx_x = 0; idx_x < x_est.size() - 1; ++idx_x) {
                std::cout << x_est[idx_x] << ", ";
                writeFile << x_est[idx_x] << ", ";
            }
            std::cout << x_est[x_est.size() - 1] << std::endl;
            writeFile << x_est[x_est.size() - 1] << std::endl;
            std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            writeFile << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
            res_max_gap = max_gap;
            max_gap = -1e10; // refresh max gap
            it_num += 1;
        } // end while (flag_terminate == false)
        
        // update x_old
        for (int x_idx = 0; x_idx < model_parameters.A.getColLength(); ++x_idx) {
            res.x[x_idx] = x_est[x_idx];
        }
        res.max_gap = res_max_gap;
        if (flag_terminate == true) { // 0: Problem is solved; 1: Problem is not solved
            res.sol_flag = 0;
        }
        else {
            res.sol_flag = 1;
        }
        res.it_num = it_num;
        // output solution at the iteration pointer
        // syntax: iteration time total_it max_gap solution
        if (idx_pointer < it_pointer.size() && it == it_pointer[idx_pointer] - 1) {
            double duration2 = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
            writeFileSol << it_pointer[idx_pointer] << "," << duration2 << "," << it_count << "," << res.max_gap;
            for (int idx_x = 0; idx_x < res.x.size(); ++idx_x) {
                writeFileSol << "," << res.x[idx_x];
            }
            writeFileSol << std::endl;
            idx_pointer++;
        }
        // === kNN Estimation ===
        // clear the memory of the previous kNN set
        
        std::cout << "===kNN Estimation===\n";
        writeFile << "===kNN Estimation===\n";
        // check if it is not the last iteration
        if (it < it_max - 1) {
            batch_cur += batch_incre;
            for (int it_data = 0; it_data < batch_cur; ++it_data) {
                idx_datapoint += 1;
                N += 1;
                k_new = pow(N, beta);
                // obtain a new data point
                dataPoint be_datapoint;
                if (flag_be == true) {
                    be_datapoint = be_DB[0][idx_datapoint];
                }
                /*
                else {
                    std::cout << "No random variable is in be of the equality constraint of the second stage problem.\n";
                }
                 */
                dataPoint bi_datapoint;
                if (flag_bi == true) {
                    bi_datapoint = bi_DB[0][idx_datapoint];
                }
                /*
                else {
                    std::cout << "No random variable is in bi of the inequality constraint of the second stage problem.\n";
                }
                 */
                dataPoint Ce_datapoint;
                if (flag_Ce == true) {
                    Ce_datapoint = Ce_DB[0][idx_datapoint];
                }
                /*
                else {
                    std::cout << "No random variable is in Ce of the equality constraint of the second stage problem.\n";
                }
                 */
                dataPoint Ci_datapoint;
                if (flag_Ci == true) {
                    Ci_datapoint = Ci_DB[0][idx_datapoint];
                }
                /*
                else {
                    std::cout << "No random variable is in Ci of the inequality constraint of the second stage problem.\n";
                }
                 */
                //*********************
                secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
                RHS_dataset.push_back(RHS_datapoint);
                // compute the distance to the observed predictor
                double distance_squared = 0;
                for (int idx_component = 0; idx_component < RHS_datapoint.predictor.size(); ++idx_component) {
                    distance_squared += (RHS_datapoint.predictor[idx_component] - observed_predictor[idx_component]) * (RHS_datapoint.predictor[idx_component] - observed_predictor[idx_component]);
                }
                distanceSet.push_back(distance_squared);
                // store the new squared distance
                // sorting (like insert sorting)
                //int left_index = 0; // the index corresponds to the largest distance that is smaller than the current one
                //double left_distance = -1;
                // double indices used for tie-breaking
                int right_index = -1; // the index corresponds to the smallest distance that is larger than  the current one
                double right_distance = -1;
                for (int index = 0; index < orderSet.size(); ++index) {
                    /*
                    if (distanceSet[index] < distance_squared) {
                        if (left_index == 0) {
                            left_distance = distanceSet[index];
                            left_index = orderSet[index];
                        }
                        else if (distanceSet[index] > left_distance) {
                            left_distance = distanceSet[index];
                            left_index = orderSet[index];
                        }
                    }
                     */
                    if (distanceSet[index] > distance_squared) {
                        if (right_index == -1) {
                            right_distance = distanceSet[index];
                            right_index = orderSet[index];
                        }
                        else if (distanceSet[index] < right_distance) {
                            right_distance = distanceSet[index];
                            right_index = orderSet[index];
                        }
                        else if (distanceSet[index] == right_distance && right_index > orderSet[index]) {
                            right_index = orderSet[index];
                        }
                    }
                }
                /*
                if (flag_debug == true) {
                    std::cout << "Output double indices\n";
                    writeFile << "Output double indices\n";
                    std::cout << "left index: " << left_index << std::endl;
                    writeFile << "left index: " << left_index << std::endl;
                    std::cout << "right index: " << right_index << std::endl;
                    writeFile << "right index: " << right_index << std::endl;
                }
                 */
                // update the orderSet
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (right_index != -1 && orderSet[index] >= right_index) {
                        orderSet[index] = orderSet[index] + 1;
                    }
                    //if (left_index == 0) { // current one is the nearest neighbor
                    //    orderSet[index] = orderSet[index] + 1;
                    //}
                    //else if (orderSet[index] > left_index) {
                    //    orderSet[index] = orderSet[index] + 1;
                    //}
                }
                if (right_index == -1) {
                    orderSet.push_back((int) orderSet.size() + 1);
                }
                else {
                    orderSet.push_back(right_index);
                }
                /*
                if (flag_debug == true) {
                    std::cout << "Updated Order in the scenario set\n";
                    writeFile << "Updated Order in the scenario set\n";
                    std::cout << "Index, Order, Distance (Squared)\n";
                    writeFile << "Index, Order, Distance (Squared)\n";
                    // update the kNN set
                    for (int index = 0; index < orderSet.size(); ++index) {
                        std::cout << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                        writeFile << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                        if (orderSet[index] <= k_new) {
                            std::cout << "*";
                            writeFile << "*";
                            kNNSet_new.push_back(index);
                        }
                        std::cout << std::endl;
                        writeFile << std::endl;
                    }
                }
                else {
                    // update the kNN set
                    for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            kNNSet_new.push_back(index);
                        }
                    }
                }
                 */
                kNNSet.clear(); // clear the old kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet.push_back(index);
                    }
                }
            } // end batch sampling
            // update k
            k = k_new;
        } // end if (it < it_max - 1)
    } // end main loop
    // end environment
    env_sub.end();
    // return estimated solutions
    std::cout << "Finish Solving Process\n";
    writeFile << "Finish Solving Process\n";
    std::cout << "Total Number of MBD Iterations: " << it_count << std::endl;
    writeFile << "Total Number of MBD Iterations: " << it_count << std::endl;
    res.mbd_count = it_count;
    std::cout << "Estimated First-Stage Solution: \n";
    writeFile << "Estimated First-Stage Solution: \n";
    for (int idx_x = 0; idx_x < res.x.size() - 1; ++idx_x) {
        std::cout << res.x[idx_x] << ", ";
        writeFile << res.x[idx_x] << ", ";
    }
    std::cout << res.x[res.x.size() - 1] << std::endl;
    writeFile << res.x[res.x.size() - 1] << std::endl;
    // write time elapsed
    double duration = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    writeFile << "Time elapsed(secs) : " << duration << "\n";
    writeFile.close();
    writeFileSol.close();
    return res;
}
