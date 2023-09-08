//
//  NSD_solver_batch.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 3/25/23.
//

#include "NSD_solver_batch.hpp"

// SD-kNN with batch samlping
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
                              bool flag_debug) {
    // prespecified precision
    const double SOLVER_PRECISION_LOWER = -1e-6;
    const double SOLVER_PRECISION_UPPER = 1e-6;
    const double SOLVER_INF = 1e10;
    
    // timer
    std::clock_t time_start;
    time_start = std::clock();
    // current time
    std::time_t currTime = std::time(nullptr);
    // STEP 1: INITIALIZATION
    // algorithm parameters
    double sigma = 1.0;
    double q = 0.5;
    //double beta = 0.5; 0.6 is okay
    double beta = 0.6; // 0 < beta < 1
    int k = 1;
    int k_new = 1;
    int N = 0;
    int N_pre = batch_init; // N_pre is used for presolving, the first iteration always start with the size with batch_init
    std::vector<double> distanceSet;
    std::vector<int> orderSet;
    std::vector<int> kNNSet;
    double knn_radius = 0;
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
    std::string resultsOutput_path = folder_path + "/computationalResults(sdknn_batch_3.1).txt";
    std::string resultsSolution_path = folder_path + "/sol(sdknn_batch_v3.1).txt";
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
    // read  be
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
    // create model structure
    standardTwoStageParameters model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
    secondStageRHSmap RHSmap = readStochasticMap(sto_path);
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
    std::cout << "SD-kNN-Batch (v3.1) is initialized\n";
    writeFile << "SD-kNN-Batch (v3.1) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, batch_init, batch_incre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, batch_init, batch_incre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << batch_init << ", " << batch_incre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << batch_init << ", " << batch_incre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
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
    long A_rowsize = model_parameters.A.getRowLength();
    long A_colsize = model_parameters.A.getColLength();
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int predictor_index = 0; predictor_index < observed_predictor.size() - 1; ++predictor_index) {
        std::cout << observed_predictor[predictor_index] << ", ";
        writeFile << observed_predictor[predictor_index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    // kNN Estimation Initialization
    std::cout << "===kNN Estimation Initialization===\n";
    writeFile << "===kNN Estimation Initialization===\n";
    // find the kNN set
    for (int idx_pre = 0; idx_pre < N_pre; ++idx_pre) {
        k_new = (int) pow(N_pre, beta);
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
    int rel_kpos = 0; // the relative position of the kth NN in the unordered kNN set
    int rel_newpos = 0; // relative position of the new NN
    std::vector<int> newkNNSet;
    int rel_idx = -1;
    for (int index = 0; index < orderSet.size(); ++index) {
        if (orderSet[index] <= k_new) {
            kNNSet.push_back(index);
            rel_idx += 1;
            if (orderSet[index] == k_new) {
                knn_radius = distanceSet[index];
                //rel_kpos = (int) kNNSet.size(); // record the relative position of the kth NN in the unordered kNN set
                rel_kpos = rel_idx;
            }
        }
    }
    // calculate kNN point estimate fro the presolving process
    secondStageRHSpoint knn_point_estimate;
    if (flag_be == true) { // be sto part exists
        // initialize point estimate
        for (int idx = 0; idx < RHS_dataset[0].be.size(); ++idx) {
            knn_point_estimate.be.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].be.size(); ++idx_component) {
                knn_point_estimate.be[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].be[idx_component];
            }
        }
    }
    if (flag_bi == true) { // bi sto part exists
        for (int idx = 0; idx < RHS_dataset[0].bi.size(); ++idx) {
            knn_point_estimate.bi.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].bi.size(); ++idx_component) {
                knn_point_estimate.bi[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].bi[idx_component];
            }
        }
    }
    if (flag_Ce == true) { // Ce sto part exists
        for (int idx = 0; idx < RHS_dataset[0].Ce.size(); ++idx) {
            knn_point_estimate.Ce.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].Ce.size(); ++idx_component) {
                knn_point_estimate.Ce[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].Ce[idx_component];
            }
        }
    }
    if (flag_Ci == true) { // Ci sto part exists
        for (int idx = 0; idx < RHS_dataset[0].Ci.size(); ++idx) {
            knn_point_estimate.Ci.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].Ci.size(); ++idx_component) {
                knn_point_estimate.Ci[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].Ci[idx_component];
            }
        }
    }
    // presolve problem to get x_incumbent
    std::vector<double> x_incumbent = nsd_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    std::vector<double> pi_e_collection;
    std::vector<std::vector<double>> minus_pi_C_collection;
    // =================================
    // set up subproblem environment
    // set up the model
    IloEnv env_sub;
    IloModel mod_sub(env_sub);
    IloNumVarArray y(env_sub,model_parameters.D.getColLength(),model_parameters.y_lb,IloInfinity,ILOFLOAT);
    mod_sub.add(y);
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
            exprs_eq_sub[model_parameters.C.getRow(idx)] += model_parameters.C.getVal(idx) * x_incumbent[col_idx];
        }
    }
    // right hand side e
    for (int idx = 0; idx < model_parameters.e.getNzeroLen(); ++idx) {
        exprs_eq_sub[model_parameters.e.getLoc(idx)] -= model_parameters.e.getVal(idx);
    }
    // coefficients before x (stochastic part) equality
    for (int idx = 0; idx < RHS_dataset[kNNSet[0]].Ce.size(); ++idx) {
        exprs_eq_sub[RHSmap.Ce_map[idx].first] += RHS_dataset[kNNSet[0]].Ce[idx] * x_incumbent[RHSmap.Ce_map[idx].second];
    }
    // coefficients before x (stochastic part) inequality (location is behind equality constraints)
    for (int idx = 0; idx < RHS_dataset[kNNSet[0]].Ci.size(); ++idx) {
        exprs_eq_sub[RHSmap.Ci_map[idx].first + model_parameters.num_eq] += RHS_dataset[kNNSet[0]].Ci[idx] * x_incumbent[RHSmap.Ci_map[idx].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx = 0; idx < RHS_dataset[kNNSet[0]].be.size(); ++idx) {
        exprs_eq_sub[RHSmap.be_map[idx]] -= RHS_dataset[kNNSet[0]].be[idx];
    }
    // right hand side (stochastic part) equality bi_(i) inequality
    for (int idx = 0; idx < RHS_dataset[kNNSet[0]].bi.size(); ++idx) {
        exprs_eq_sub[RHSmap.bi_map[idx] + model_parameters.num_eq] -= RHS_dataset[kNNSet[0]].bi[idx];
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
    // det C x
    for (int col_idx = 0; col_idx < model_parameters.C.getColLength(); ++col_idx) {
        int beg_idx = model_parameters.C.getCbeg(col_idx);
        for (int idx = beg_idx; idx < model_parameters.C.getClen(col_idx) + beg_idx; ++idx) {
            det_rhs_bounds[model_parameters.C.getRow(idx)] -= model_parameters.C.getVal(idx) * x_incumbent[col_idx];
        }
    }
    std::vector<double> rhs_bounds(model_parameters.D.getRowLength(), 0.0);
    // ================================= (End of Subproblem Setup)
    // *********************************
    // intermediate variables for computation
    std::vector<double> dual_workingVector(model_parameters.D.getRowLength(),0.0);
    double l1_norm = 0;
    bool flag_new = true;
    // *********************************
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        // update the deterministic part
        for (int idx_row = 0; idx_row < model_parameters.D.getRowLength(); ++idx_row) {
            rhs_bounds[idx_row] = det_rhs_bounds[idx_row];
        }
        // update the stochastic parts of e
        for (int idx_be = 0; idx_be < RHS_dataset[kNNSet[idx_knn]].be.size(); ++idx_be) {
            rhs_bounds[RHSmap.be_map[idx_be]] += RHS_dataset[kNNSet[idx_knn]].be[idx_be];
        }
        // right hand side (stochastic part) equality bi_(i) inequality
        for (int idx_bi = 0; idx_bi < RHS_dataset[kNNSet[idx_knn]].bi.size(); ++idx_bi) {
            rhs_bounds[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] += RHS_dataset[kNNSet[idx_knn]].bi[idx_bi];
        }
        // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
        for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[idx_knn]].Ce.size(); ++idx_Ce) {
            rhs_bounds[RHSmap.Ce_map[idx_Ce].first] -= RHS_dataset[kNNSet[idx_knn]].Ce[idx_Ce] * x_incumbent[RHSmap.Ce_map[idx_Ce].second];
        }
        // coefficients before x (stochastic part) inequality (location is behind equality constraints)
        for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[idx_knn]].Ci.size(); ++idx_Ci) {
            rhs_bounds[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] -= RHS_dataset[kNNSet[idx_knn]].Ci[idx_Ci] * x_incumbent[RHSmap.Ci_map[idx_Ci].second];
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
            //double optimal_value = cplex_sub.getObjValue(); // get the optimal value
            cplex_sub.getDuals(dual_equality_sub,constraintsEquality_sub);
            l1_norm = 0; // reset l1_norm
            for (int index_eq = 0; index_eq < model_parameters.D.getRowLength(); ++index_eq) {
                double pi_temp = dual_equality_sub[index_eq]; // move y to the right hand side
                dual_workingVector[index_eq] = pi_temp;
                l1_norm += abs(pi_temp);
            }
            //std::cout << "l1 norm: " << l1_norm << std::endl;
            //for (int idx = 0; idx < dual_workingVector.size(); ++idx) {
            //    std::cout << dual_workingVector[idx] << " ";
            //}
            //std::cout << std::endl;
            // check if l1_norm is new
            flag_new = true;
            for (int idx_dual = 0; idx_dual < explored_duals.size(); ++idx_dual) {
                double diff = l1_norm - explored_duals[idx_dual].l1_norm;
                if (diff >= SOLVER_PRECISION_LOWER && diff <= SOLVER_PRECISION_UPPER) { // l1 norm is not new
                    flag_new = false;
                    break;
                }
            }
            //flag_new = true; // for debug
            if (flag_new == true) { // find a new dual extreme point
                dualMultipliers curr;
                curr.dual = dual_workingVector;
                curr.l1_norm = l1_norm;
                explored_duals.push_back(curr);
                // deterministic e
                double pi_e = model_parameters.e.fast_dotProduct(curr.dual);
                pi_e_collection.push_back(pi_e);
                // determinictic C
                std::vector<double> pi_C = model_parameters.C.fast_rightMultiply(curr.dual);
                // negate
                for (int idx = 0; idx < pi_C.size(); ++idx) {
                    pi_C[idx] = pi_C[idx] * (-1.0);
                }
                minus_pi_C_collection.push_back(pi_C);
            }
        } // end if (flag_solve == IloTrue)
    } // end for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn)
    for (int index_eq = 0; index_eq < model_parameters.D.getRowLength(); ++index_eq) { // erase working vector
        dual_workingVector[index_eq] = 0;
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant
    for (int idx_x = 0; idx_x < model_parameters.A.getColLength(); ++idx_x) {
        initial_minorant.beta.push_back(0);
    }
    minorant_collection.push_back(initial_minorant);
    std::cout << "Maximum number of iterations: " << max_iterations << std::endl;
    writeFile << "Maximum number of iterations: " << max_iterations << std::endl;
    // main loop
    std::cout << "Start Solving Process\n";
    writeFile << "Start Solving Process\n";
    // initialize the index for the datapoint
    int idx_datapoint = batch_init - 1;
    int batch = batch_init;
    N = batch_init; // update number of data points collected
    k = k_new;
    bool flag_first_update = true;
    // intermediate values for the solver
    std::vector<double> old_candidate = x_incumbent;
    std::vector<double> old_incumbent = x_incumbent;
    minorant old_minorant_incumbent = initial_minorant;
    minorant old_minorant_candidate = initial_minorant;
    double old_knn_radius = knn_radius;
    std::vector<double> x_candidate(A_colsize, 0.0);
    //std::vector<minorant> new_minorant_collection;
    //std::vector<double> minorant_l1norm;
    std::vector<double> minus_pi_C_candidate;
    std::vector<double> minus_pi_C_incumbent;
    // main loop
    int idx_pointer = 0;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::cout << "***Iteration " << iteration << "***\n";
        writeFile << "***Iteration " << iteration << "***\n";
        std::cout << "sigma: " << sigma << std::endl;
        writeFile << "sigma: " << sigma << std::endl;
        double half_sigma = sigma * 0.5;
        // ****************
        // Candidate Selection
        //std::cout << "===CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        //writeFile << "===CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        // solve master problem with a proximal term
        IloEnv env;
        IloModel mod(env);
        IloNumVarArray x_temp(env,A_colsize,model_parameters.x_lb,IloInfinity,ILOFLOAT);
        IloNumVar eta(env,f_lowerbound,IloInfinity,ILOFLOAT);
        //IloNumVar eta(env,-IloInfinity,IloInfinity,ILOFLOAT);
        mod.add(x_temp);
        mod.add(eta);
        IloExpr expr_obj(env);
        for (int idx = 0; idx < model_parameters.c.getNzeroLen(); ++idx) {
            expr_obj += model_parameters.c.getVal(idx) * x_temp[model_parameters.c.getLoc(idx)];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += half_sigma * x_temp[x_index] * x_temp[x_index] - sigma * x_incumbent[x_index] * x_temp[x_index];
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
        std::vector<IloExpr> exprs_regular;
        for (int index_row = 0; index_row < A_rowsize; ++index_row) {
            IloExpr expr(env);
            exprs_regular.push_back(expr);
        }
        // Ax <= b
        for (int col_idx = 0; col_idx < model_parameters.A.getColLength(); ++col_idx) {
            int beg_idx = model_parameters.A.getCbeg(col_idx);
            for (int idx = beg_idx; idx < model_parameters.A.getClen(col_idx) + beg_idx; ++idx) {
                exprs_regular[model_parameters.A.getRow(idx)] += model_parameters.A.getVal(idx) * x_temp[col_idx];
            }
        }
        // right hand side
        for (int idx = 0; idx < model_parameters.b.getNzeroLen(); ++idx) {
            exprs_regular[model_parameters.b.getLoc(idx)] -= model_parameters.b.getVal(idx);
        }
        // add constraints
        for (int idx = 0; idx < model_parameters.A.getRowLength(); ++idx) {
            mod.add(exprs_regular[idx] <= 0);
        }
        // constrants for minorants
        IloRangeArray minorant_constraints(env);
        std::cout << "Number of minorants used in the regularized master problem: " << minorant_collection.size() << std::endl;
        writeFile << "Number of minorants used in the regularized master problem: " << minorant_collection.size() << std::endl;
        for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
            IloExpr expr(env);
            expr += minorant_collection[idx_minorant].alpha - eta;
            for (int idx_x = 0; idx_x < A_colsize; ++idx_x) {
                expr += minorant_collection[idx_minorant].beta[idx_x] * x_temp[idx_x];
            }
            minorant_constraints.add(expr <= 0);
            //std::cout << "Debug: idx_minorant = " << idx_minorant << std::endl;
        }
        mod.add(minorant_constraints);
        // constraints for feasibility (won't appear in the current version)
        // create cplex environment
        IloCplex cplex(env);
        cplex.extract(mod);
        cplex.setOut(env.getNullStream());
        //cplex.exportModel("/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/master.lp");
        cplex.solve();
        //std::cout << "Debug: Solved\n";
        // obtain the proximal point (condidate solution)
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            x_candidate[x_index] = cplex.getValue(x_temp[x_index]);
            //std::cout << cplex.getValue(x_temp[x_index]) << std::endl;
        }
        // obtain the duals
        IloNumArray duals(env);
        cplex.getDuals(duals, minorant_constraints);
        // compute RHS of incumbent selection
        // value at the candidate
        double f_old_candidate = 0;
        // value at the incumbent
        double f_old_incumbent = 0;
        for (int idx = 0; idx < model_parameters.c.getNzeroLen(); ++idx) {
            f_old_candidate += model_parameters.c.getVal(idx) * x_candidate[model_parameters.c.getLoc(idx)];
            f_old_incumbent += model_parameters.c.getVal(idx) * x_incumbent[model_parameters.c.getLoc(idx)];
        }
        double f_new_candidate = f_old_candidate;
        double f_new_incumbent = f_old_incumbent;
        f_old_candidate += cplex.getValue(eta);
        // second stage value at the incumbent
        double recourse_incumbent = f_lowerbound;
        for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
            double piece_val = minorant_collection[idx_minorant].alpha;
            piece_val += minorant_collection[idx_minorant].beta * x_incumbent;
            if (piece_val > recourse_incumbent) {
                recourse_incumbent = piece_val;
            }
        }
        f_old_incumbent += recourse_incumbent;
        // *** only store unique minorants
        int dual_idx = -1;
        for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
            dual_idx++;
            if (duals[dual_idx] >= SOLVER_PRECISION_LOWER && duals[dual_idx] <= SOLVER_PRECISION_UPPER) { // remove inactive minorants
                minorant_collection.erase(minorant_collection.begin() + idx_minorant);
                idx_minorant--;
            }
            /*
            if (duals[dual_idx] < SOLVER_PRECISION_LOWER || duals[dual_idx] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
                
                bool flag_newm = true;
                double ml1norm = 0;
                for (int idx = 0; idx < minorant_collection[idx_minorant].beta.size(); ++idx) {
                    ml1norm += abs(minorant_collection[idx_minorant].beta[idx]);
                }
                ml1norm += abs(minorant_collection[idx_minorant].alpha);
                
                for (int idx = 0; idx < minorant_l1norm.size(); ++idx) {
                    double m_diff = ml1norm - minorant_l1norm[idx];
                    if (m_diff < SOLVER_PRECISION_UPPER && m_diff > SOLVER_PRECISION_LOWER) {
                        flag_new = false; // minorant is not new
                        std::cout << "*** Active Minorant is not unique ***\n";
                        break;
                    }
                }
                if (flag_newm == true) {
                    minorant_l1norm.push_back(ml1norm);
                    //new_minorant_collection.push_back(minorant_collection[idx_minorant]);
                }
                else { // delete the active non-unique minorant
                    minorant_collection.erase(minorant_collection.begin() + idx_minorant);
                    idx_minorant--;
                }
                 
            }
            else { // delete inactive minorant
                minorant_collection.erase(minorant_collection.begin() + idx_minorant);
                idx_minorant--;
            }
             */
        }
        //for (int idx = 0; idx < minorant_l1norm.size(); ++idx) {
        //    std::cout << minorant_l1norm[idx] << " ";
        //}
        //std::cout << std::endl;
        /*
        std::cout << "Print out active minorants:\n";
        for (int idx = 0; idx < minorant_collection.size(); ++idx) {
            std::cout << "alpha: " << minorant_collection[idx].alpha << std::endl;
            std::cout << "beta: " << std::endl;
            for (int idx2 = 0; idx2 < minorant_collection[idx].beta.size(); ++idx2) {
                std::cout << minorant_collection[idx].beta[idx2] << " ";
            }
            std::cout << std::endl;
        }
         */
        // end the cplex environment
        env.end();
        // output candidate solution
        std::cout << "Candidate Solution: ";
        writeFile << "Candidate Solution: ";
        for (int x_index = 0; x_index < A_colsize - 1; ++x_index) {
            std::cout << x_candidate[x_index] << ", ";
            writeFile << x_candidate[x_index] << ", ";
        }
        std::cout << x_candidate[A_colsize - 1] << std::endl;
        writeFile << x_candidate[A_colsize - 1] << std::endl;
        //std::cout << "===(END) CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        //writeFile << "===(END) CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        // ****************
        //*********************
        // kNN ESTIMATION
        //non-parametric estimation (kNN)
        // calculate distance squared
        //std::cout << "===kNN ESTIMATION===\n";
        //writeFile << "===kNN ESTIMATION===\n";
        std::cout << "Start batch sampling\n";
        writeFile << "Start batch sampling\n";
        batch += batch_incre;
        bool flag_kNN_change = false;
        for (int idx_bacth_datapoint = 0; idx_bacth_datapoint < batch; ++idx_bacth_datapoint) {
            idx_datapoint += 1; // go to the next data point
            N += 1;
            k_new = (int) pow(N, beta); // calculate new k
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
            //*********************
            // kNN ESTIMATION
            //non-parametric estimation (kNN)
            // calculate distance squared
            //std::cout << "===kNN ESTIMATION===\n";
            //writeFile << "===kNN ESTIMATION===\n";
            secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
            RHS_dataset.push_back(RHS_datapoint);
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
            //bool flag_kNN_change;
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
            } // end for (int index = 0; index < orderSet.size(); ++index)
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
            // update the kNN set
            if (k == k_new && old_knn_radius <= distance_squared) { // kNN set radius is not changed, new one does not enter the kNN set
                std::cout << "kNN set does not change.\n";
                flag_kNN_change = false;
            }
            else { // kNN set is changed, new one enters the kNN set
                flag_kNN_change = true;
                kNNSet.clear(); // clear the old kNN set
                newkNNSet.clear();
                // *remove the intermediate value for the duals, if k does not increase*
                if (k == k_new && flag_first_update == false) {
                    for (int idx = 0; idx < explored_duals.size(); ++idx) {
                        explored_duals[idx].sto_pi_e.erase(explored_duals[idx].sto_pi_e.begin() + rel_kpos); // clear the memory of the pi_e for the old kth NN
                    }
                    if (flag_debug == true) {
                        std::cout << "The memory of the pi_e for the old kth NN is cleared\n";
                    }
                }
                bool find_new = false;
                int count_kNN = 0;
                int rel_idx = -1;
                if (k == k_new) {
                    rel_newpos = k - 1; // new one is the last one
                    for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            count_kNN += 1;
                            rel_idx += 1;
                            //newkNNSet.push_back(index);
                            kNNSet.push_back(index);
                            if (orderSet[index] == k_new) {
                                knn_radius = distanceSet[index]; // record the radius of the kNN set
                                rel_kpos = rel_idx; // record the relative position of the kth NN in the unordered kNN set
                            }
                        }
                    }
                }
                else { // k increases by one, the new NN can be the newly generated one or the one that is already generated
                    for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            count_kNN += 1;
                            rel_idx += 1;
                            newkNNSet.push_back(index);
                            if (find_new == false && count_kNN == k_new) {
                                rel_newpos = rel_idx;
                                find_new = true;
                            }
                            else if (find_new == false && newkNNSet[rel_idx] != kNNSet[rel_idx]) {
                                rel_newpos = rel_idx;
                                find_new = true;
                            }
                            if (orderSet[index] == k_new) {
                                knn_radius = distanceSet[index]; // record the radius of the kNN set
                                rel_kpos = rel_idx; // record the relative position of the kth NN in the unordered kNN set
                            }
                        }
                    }
                    // update kNN set
                    kNNSet = newkNNSet;
                }
            } // end if (k == k_new && old_knn_radius <= distance_squared)
            if (flag_debug == true) {
                std::cout << "Debug: explored_duals[0].sto_pi_e.size() = " << explored_duals[0].sto_pi_e.size() << std::endl;
                std::cout << "Debug: rel_kpos = " << rel_kpos << std::endl;
                std::cout << "Debug: rel_newpos = " << rel_newpos << std::endl;
                for (int idx = 0; idx < kNNSet.size(); ++idx) {
                    std::cout << "kNNSet[" << idx << "] = " << kNNSet[idx] << std::endl;
                }
            }
            // compute a new dual at x_candidate
            det_rhs_bounds = e_det;
            // det C x
            for (int col_idx = 0; col_idx < model_parameters.C.getColLength(); ++col_idx) {
                int beg_idx = model_parameters.C.getCbeg(col_idx);
                for (int idx = beg_idx; idx < model_parameters.C.getClen(col_idx) + beg_idx; ++idx) {
                    det_rhs_bounds[model_parameters.C.getRow(idx)] -= model_parameters.C.getVal(idx) * x_candidate[col_idx];
                }
            }
            // update the subproblem
            for (int idx_row = 0; idx_row < model_parameters.D.getRowLength(); ++idx_row) {
                rhs_bounds[idx_row] = det_rhs_bounds[idx_row];
            } // update the deterministic part
            // update the stochastic parts of e
            for (int idx_be = 0; idx_be < RHS_datapoint.be.size(); ++idx_be) {
                rhs_bounds[RHSmap.be_map[idx_be]] += RHS_datapoint.be[idx_be];
            }
            // right hand side (stochastic part) equality bi_(i) inequality
            for (int idx_bi = 0; idx_bi < RHS_datapoint.bi.size(); ++idx_bi) {
                rhs_bounds[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] += RHS_datapoint.bi[idx_bi];
            }
            // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
            for (int idx_Ce = 0; idx_Ce < RHS_datapoint.Ce.size(); ++idx_Ce) {
                rhs_bounds[RHSmap.Ce_map[idx_Ce].first] -= RHS_datapoint.Ce[idx_Ce] * x_candidate[RHSmap.Ce_map[idx_Ce].second];
            }
            // coefficients before x (stochastic part) inequality (location is behind equality constraints)
            for (int idx_Ci = 0; idx_Ci < RHS_datapoint.Ci.size(); ++idx_Ci) {
                rhs_bounds[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] -= RHS_datapoint.Ci[idx_Ci] * x_candidate[RHSmap.Ci_map[idx_Ci].second];
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
                //double optimal_value = cplex_sub.getObjValue(); // get the optimal value
                cplex_sub.getDuals(dual_equality_sub,constraintsEquality_sub);
                l1_norm = 0; // reset l1_norm
                for (int index_eq = 0; index_eq < model_parameters.D.getRowLength(); ++index_eq) {
                    double pi_temp = dual_equality_sub[index_eq]; // move y to the right hand side
                    dual_workingVector[index_eq] = pi_temp;
                    l1_norm += abs(pi_temp);
                }
                // check if l1_norm is new
                flag_new = true;
                for (int idx_dual = 0; idx_dual < explored_duals.size(); ++idx_dual) {
                    double diff = l1_norm - explored_duals[idx_dual].l1_norm;
                    if (diff > SOLVER_PRECISION_LOWER && diff < SOLVER_PRECISION_UPPER) { // l1 norm is not new
                        flag_new = false;
                        break;
                    }
                }
                if (flag_new == true) { // find a new dual extreme point
                    dualMultipliers curr;
                    curr.dual = dual_workingVector;
                    curr.l1_norm = l1_norm;
                    explored_duals.push_back(curr);
                    // deterministic e
                    double pi_e = model_parameters.e.fast_dotProduct(curr.dual);
                    pi_e_collection.push_back(pi_e);
                    // determinictic C
                    std::vector<double> pi_C = model_parameters.C.fast_rightMultiply(curr.dual);
                    // negate
                    for (int idx = 0; idx < pi_C.size(); ++idx) {
                        pi_C[idx] = pi_C[idx] * (-1.0);
                    }
                    minus_pi_C_collection.push_back(pi_C);
                }
            } // end if (flag_solve == IloTrue)
            else {
                throw std::logic_error("Main solving process terminate. Subproblem is infeasible.\n");
            }
            // erase working vector
            for (int index_eq = 0; index_eq < model_parameters.D.getRowLength(); ++index_eq) {
                dual_workingVector[index_eq] = 0;
            }
            std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
            writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
            //std::cout << "===(END) DUAL SPACE EXPLORATION===\n";
            //writeFile << "===(END) DUAL SPACE EXPLORATION===\n";
            
            // MINORANT UPDATES
            // update old minorants
            //std::cout << "Update old active minorants.\n";
            //writeFile << "Update old active minorants.\n";
            //minorant_collection.clear(); // remove all the previous minorants
            if (k == k_new) {
                if (old_knn_radius - knn_radius > SOLVER_PRECISION_LOWER && old_knn_radius- knn_radius < SOLVER_PRECISION_UPPER) { // knn set does not change
                    std::cout << "kNN set does not change, old active minorants will not be updated.\n";
                    /*
                    for (int idx_minorant = 0; idx_minorant < new_minorant_collection.size(); ++idx_minorant) {
                        minorant_collection.push_back(new_minorant_collection[idx_minorant]);
                    }
                     */
                }
                else{ // knn set changes
                    std::cout << "kNN set changes.\n";
                    /*
                    for (int idx_minorant = 0; idx_minorant < new_minorant_collection.size(); ++idx_minorant) {
                        minorant tmp_minorant;
                        tmp_minorant.alpha = new_minorant_collection[idx_minorant].alpha + (f_lowerbound - f_upperbound) / ((double) k);
                        tmp_minorant.beta = new_minorant_collection[idx_minorant].beta;
                        minorant_collection.push_back(tmp_minorant);
                    }
                     */
                    for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
                        minorant_collection[idx_minorant].alpha = minorant_collection[idx_minorant].alpha + (f_lowerbound - f_upperbound) / ((double) k);
                    }
                }
                /*
                for (int idx_minorant = 0; idx_minorant < new_minorant_collection.size(); ++idx_minorant) {
                    minorant tmp_minorant;
                    tmp_minorant.alpha = new_minorant_collection[idx_minorant].alpha + (f_lowerbound - f_upperbound) / ((double) k);
                    tmp_minorant.beta = new_minorant_collection[idx_minorant].beta;
                    minorant_collection.push_back(tmp_minorant);
                }
                 */
            }
            else {
                double k_over_k_new = ((double) k) / ((double) k_new);
                double k_new_minus_k_over_k_new = ((double)(k_new - k)) / ((double) k_new);
                for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
                    /*
                    minorant tmp_minorant;
                    tmp_minorant.alpha = new_minorant_collection[idx_minorant].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                    tmp_minorant.beta = ((double) k) / ((double) k_new) * new_minorant_collection[idx_minorant].beta;
                    minorant_collection.push_back(tmp_minorant);
                    */
                    minorant_collection[idx_minorant].alpha = minorant_collection[idx_minorant].alpha * k_over_k_new + k_new_minus_k_over_k_new * f_lowerbound;
                    for (int idx = 0; idx < minorant_collection[idx_minorant].beta.size(); ++idx) {
                        minorant_collection[idx_minorant].beta[idx] = k_over_k_new * minorant_collection[idx_minorant].beta[idx];
                    }
                }
            } // end minorant updates
            // ***END MINORANT UPDATES***
            
            // store itermediate values in kNN set except last iteration,
            // values in the last iteration will be stored in the minorant construction step
            if (idx_bacth_datapoint < batch - 1) {
                for (int index = 0; index < k_new; ++index) { //
                    for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                        // pi_e - pi_C
                        // deterministic e
                        // inequality part (before standardizing) inequality constraint is after the equality constraints
                        if (flag_first_update == true) { // first iteration
                            double sto_pi_e = 0;
                            for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                            }
                            for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                            }
                            explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                        }
                        else { // not the first iteration
                            if (flag_kNN_change == true) { // kNN set changes
                                if (flag_new == false) { // dual set is not changed
                                    // only update the entries for the new NN
                                    if (index == rel_newpos) {
                                        double sto_pi_e = 0;
                                        for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                            sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                                        }
                                        for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                            sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                                        }
                                        if (index == k_new - 1) {
                                            explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                                        }
                                        else {
                                            explored_duals[dual_index].sto_pi_e.insert(explored_duals[dual_index].sto_pi_e.begin() + index,  sto_pi_e);
                                        }
                                    }
                                } // end if (flag_new == false)
                                else { // new dual is added and kNN set is new
                                    if (dual_index == explored_duals.size() - 1) { // for new dual
                                        double sto_pi_e = 0;
                                        for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                            sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                                        }
                                        for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                            sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                                        }
                                        explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                                    }
                                    else { // for old dual
                                        if (index == rel_newpos) {
                                            double sto_pi_e = 0;
                                            for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                                            }
                                            for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                                            }
                                            if (index == k_new - 1) {
                                                explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                                            }
                                            else {
                                                explored_duals[dual_index].sto_pi_e.insert(explored_duals[dual_index].sto_pi_e.begin() + index,  sto_pi_e);
                                            }
                                        }
                                    }
                                } // else
                            } // end if (flag_kNN_change == true)
                            else if (flag_new == true && dual_index == explored_duals.size() - 1) {
                                // kNN set is old, but new dual is found
                                double sto_pi_e = 0;
                                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                    sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                                }
                                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                    sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                                }
                                explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                            }
                        } // end check if (iteration == 0)
                    } // end for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index)
                } // end for (int index = 0; index < k_new; ++index)
                // print out pi_e
                if (flag_debug == true) {
                    for (int dual_idx = 0; dual_idx < explored_duals.size(); ++dual_idx) {
                        std::cout << "dual[" << dual_idx << "](sto_pi_e): ";
                        for (int knn_idx = 0; knn_idx < k_new; ++knn_idx) {
                            std::cout << explored_duals[dual_idx].sto_pi_e[knn_idx] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
                }
            } // end if (idx_bacth_datapoint < batch - 1)
            // update k and knn radius
            k = k_new;
            old_knn_radius = knn_radius;
            flag_first_update = false;
        } // for (int idx_bacth_datapoint = 0; idx_bacth_datapoint < batch; ++idx_bacth_datapoint)
        //std::cout << "Debug flag_kNN_change: " << flag_kNN_change << std::endl;
        //***batch sampling***
        //  MINORANT CUTS CONSTRUCTION
        //std::cout << "===MINORANT CONSTRUCTION===\n";
        //writeFile << "===MINORANT CONSTRUCTION===\n";
        // find the duals correspond to the kNN
        double one_over_k_new = (1.0 / (double) k_new);
        minorant minorant_candidate;
        minorant minorant_incumbent;
        minorant_candidate.alpha = 0;
        minorant_incumbent.alpha = 0;
        for (int index_x = 0; index_x < A_colsize; ++index_x) {
            minorant_candidate.beta.push_back(0.0);
            minorant_incumbent.beta.push_back(0.0);
        }
        // intermediate value for pi' C x
        //std::vector<double> minus_pi_C_candidate;
        //std::vector<double> minus_pi_C_incumbent;
        minus_pi_C_candidate.clear();
        for (int index = 0; index < k_new; ++index) {
            double max_value = -SOLVER_INF;
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate(A_colsize,0);
            std::vector<double> beta_incumbent(A_colsize,0);
            // incumbent
            double max_value_incumbent = -SOLVER_INF;
            //std::cout << "Debug: Initial max_val = " << max_value_incumbent << std::endl;
            for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                // pi_e - pi_C
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                //double pi_e = model_parameters.e * explored_duals[dual_index].equality; // (IMPORTANT) need to allocate space to store it
                double pi_e = pi_e_collection[dual_index];
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                if (flag_kNN_change == true) { // kNN set changes
                    if (flag_new == false) { // dual set is not changed
                        // only update the entries for the new NN
                        if (index == rel_newpos) {
                            double sto_pi_e = 0;
                            for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                            }
                            for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                            }
                            if (index == k_new - 1) {
                                explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                            }
                            else {
                                explored_duals[dual_index].sto_pi_e.insert(explored_duals[dual_index].sto_pi_e.begin() + index,  sto_pi_e);
                            }
                        }
                    } // end if (flag_new == false)
                    else { // new dual is added and kNN set is new
                        if (dual_index == explored_duals.size() - 1) { // for new dual
                            double sto_pi_e = 0;
                            for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                            }
                            for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                            }
                            explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                        }
                        else { // for old dual
                            if (index == rel_newpos) {
                                double sto_pi_e = 0;
                                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                                    sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                                }
                                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                                    sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                                }
                                if (index == k_new - 1) {
                                    explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                                }
                                else {
                                    explored_duals[dual_index].sto_pi_e.insert(explored_duals[dual_index].sto_pi_e.begin() + index,  sto_pi_e);
                                }
                            }
                        }
                    } // else { // new dual is added
                } // end if (flag_kNN_change == true)
                else if (flag_new == true && dual_index == explored_duals.size() - 1) {
                    // kNN set is old, but new dual is found
                    double sto_pi_e = 0;
                    for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                        sto_pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                    }
                    for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                        sto_pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                    }
                    explored_duals[dual_index].sto_pi_e.push_back(sto_pi_e);
                }
                pi_e += explored_duals[dual_index].sto_pi_e[index];
                /*
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[dual_index].dual[RHSmap.be_map[idx_eq]] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[dual_index].dual[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                 */
                current_value += pi_e;
                // determinitsic C
                //std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C; // (IMPORTANT) need to allocate space to store it
                //std::vector<double> pi_C = pi_C_collection[dual_index];
                //SparseVector minus_pi_C = minus_pi_C_collection[dual_index];
                std::vector<double> sto_minus_pi_C(model_parameters.C.getColLength(), 0.0);
                // stochastic C
                // equality
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    sto_minus_pi_C[RHSmap.Ce_map[idx_Ce].second] -= RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].dual[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    sto_minus_pi_C[RHSmap.Ci_map[idx_Ci].second] -= RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].dual[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                //current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = pi_e;
                // deterministic part is only calculated at the first time
                if (index < 1) {
                    minus_pi_C_candidate.push_back(minus_pi_C_collection[dual_index] * x_candidate);
                    if (iteration < 1) { // first iteration
                        minus_pi_C_incumbent.push_back(minus_pi_C_collection[dual_index] * x_incumbent);
                    }
                    else {
                        if (flag_new == true && dual_index == explored_duals.size() - 1) {
                            minus_pi_C_incumbent.push_back(minus_pi_C_collection[dual_index] * x_incumbent);
                        }
                    }
                }
                current_value += minus_pi_C_candidate[dual_index];
                current_value_incumbent +=  minus_pi_C_incumbent[dual_index];
                if (flag_Ce == true || flag_Ci == true) {
                    current_value += sto_minus_pi_C * x_candidate;
                    current_value_incumbent += sto_minus_pi_C * x_incumbent;
                }
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    //beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    if (flag_Ce == true || flag_Ci == true) {
                        beta_candidate = sto_minus_pi_C;
                        beta_incumbent = sto_minus_pi_C;
                    }
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        //beta_candidate = (-1.0) * pi_C;
                        if (flag_Ce == true || flag_Ci == true) {
                            beta_candidate = sto_minus_pi_C;
                        }
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        //beta_incumbent = (-1.0) * pi_C;
                        // only store the stochastic part, det part is stored as dual_index
                        if (flag_Ce == true || flag_Ci == true) {
                            beta_incumbent = sto_minus_pi_C;
                        }
                    }
                }
            } // end for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index)
            /*
            std::cout << "max_value_incumbent: " << max_value_incumbent << std::endl;
            std::cout << "max_index_incumbent: " << max_index_incumbent << std::endl;
            for (int idx = 0; idx < beta_incumbent.size(); ++idx) {
                std::cout << beta_incumbent[idx] << " ";
            }
            std::cout << std::endl;
            std::cout << "max_value: " << max_value << std::endl;
            std::cout << "max_index: " << max_index << std::endl;
            for (int idx = 0; idx < beta_candidate.size(); ++idx) {
                std::cout << beta_candidate[idx] << " ";
            }
            std::cout << std::endl;
             */
            
            // minorant on the candidate
            minorant_candidate.alpha += alpha_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += alpha_incumbent;
            if (flag_Ce == true || flag_Ci == true) {
                minorant_candidate.beta = minorant_candidate.beta + beta_candidate;
                minorant_candidate.beta = minorant_candidate.beta + minus_pi_C_collection[max_index];
                minorant_incumbent.beta = minorant_incumbent.beta + beta_incumbent;
                minorant_incumbent.beta = minorant_incumbent.beta + minus_pi_C_collection[max_index_incumbent];
            }
            else {
                minorant_candidate.beta = minorant_candidate.beta + minus_pi_C_collection[max_index];
                minorant_incumbent.beta = minorant_incumbent.beta + minus_pi_C_collection[max_index_incumbent];
            }
        } // end for (int index = 0; index < k_new; ++index)
        
        minorant_candidate.alpha *= one_over_k_new;
        minorant_candidate.beta = one_over_k_new * minorant_candidate.beta;
        minorant_incumbent.alpha *= one_over_k_new;
        minorant_incumbent.beta = one_over_k_new * minorant_incumbent.beta;
        // print out pi_e
        if (flag_debug == true) {
            for (int dual_idx = 0; dual_idx < explored_duals.size(); ++dual_idx) {
                std::cout << "dual[" << dual_idx << "](sto_pi_e): ";
                for (int knn_idx = 0; knn_idx < k_new; ++knn_idx) {
                    std::cout << explored_duals[dual_idx].sto_pi_e[knn_idx] << " ";
                }
                std::cout << std::endl;
            }
        }
        // (END) MINORANT CUTS CONSTRUCTION
        // add new minorants
        minorant_collection.push_back(minorant_candidate);
        minorant_collection.push_back(minorant_incumbent);
        // output new minorants
        if (flag_debug == true) { //
            std::cout << "Minorant Candidate\n";
            writeFile << "Minorant Candidate\n";
            std::cout << "alpha: " << minorant_candidate.alpha << std::endl;
            writeFile << "alpha: " << minorant_candidate.alpha << std::endl;
            std::cout << "beta: ";
            writeFile << "beta: ";
            for (int x_index = 0; x_index < A_colsize - 1; ++x_index) {
                std::cout << minorant_candidate.beta[x_index] << ", ";
                writeFile << minorant_candidate.beta[x_index] << ", ";
            }
            std::cout << minorant_candidate.beta[A_colsize - 1] << std::endl;
            writeFile << minorant_candidate.beta[A_colsize - 1] << std::endl;
            std::cout << "Minorant Incumbent\n";
            writeFile << "Minorant Incumbent\n";
            std::cout << "alpha: " << minorant_incumbent.alpha << std::endl;
            writeFile << "alpha: " << minorant_incumbent.alpha << std::endl;
            std::cout << "beta: ";
            writeFile << "beta: ";
            for (int x_index = 0; x_index < A_colsize - 1; ++x_index) {
                std::cout << minorant_incumbent.beta[x_index] << ", ";
                writeFile << minorant_incumbent.beta[x_index] << ", ";
            }
            std::cout << minorant_incumbent.beta[A_colsize - 1] << std::endl;
            writeFile << minorant_incumbent.beta[A_colsize - 1] << std::endl;
        }
        //std::cout << "===(END) MINORANT CONSTRUCTION===\n";
        //writeFile << "===(END) MINORANT CONSTRUCTION===\n";
        // Incumbent Selection
        //std::cout << "===INCUMBENT SELECTION===\n";
        //writeFile << "===INCUMBENT SELECTION===\n";
        // LHS
        // new function value at candidate solution
        // second stage value
        double new_recourse_candidate = f_lowerbound;
        double new_recourse_incumbent = f_lowerbound;
        for (int idx_minorant = 0; idx_minorant < minorant_collection.size(); ++idx_minorant) {
            double piece_val = minorant_collection[idx_minorant].alpha;
            piece_val += minorant_collection[idx_minorant].beta * x_candidate;
            if (piece_val > new_recourse_candidate) {
                new_recourse_candidate = piece_val;
            }
            double piece_val2 = minorant_collection[idx_minorant].alpha;
            piece_val2 += minorant_collection[idx_minorant].beta * x_incumbent;
            if (piece_val2 > new_recourse_incumbent) {
                new_recourse_incumbent = piece_val2;
            }
        }
        f_new_candidate += new_recourse_candidate;
        f_new_incumbent += new_recourse_incumbent;
        double LHS = f_new_candidate - f_new_incumbent;
        double RHS = q * (f_old_candidate - f_old_incumbent);
        std::cout << "LHS: " << LHS << std::endl;
        writeFile << "LHS: " << LHS << std::endl;
        std::cout << "RHS: " << RHS << std::endl;
        writeFile << "RHS: " << RHS << std::endl;
        if (LHS <= RHS) {
            std::cout << "Computation Log: Incumbent selection criterion is passed.\n";
            writeFile <<"Computation Log: Incumbent selection criterion is passed.\n";
            x_incumbent = x_candidate;
            // update stepsize
            sigma = max(sigma * 0.5, sigma_lowerbound);
            // update minus_pi_C_inncumbent
            minus_pi_C_incumbent = minus_pi_C_candidate;
        }
        else {
            std::cout << "Computation Log: Incumbent selection criterion is not passed.\n";
            writeFile <<"Computation Log: Incumbent solution selection criterion is not passed.\n";
            sigma = min(sigma * 2.0, sigma_upperbound);
        }
        // print out the incumbent solution
        std::cout << "Incumbent Solution: ";
        writeFile << "Incumbent Solution: ";
        for (int index = 0; index < A_colsize -1; ++index) {
            std::cout << x_incumbent[index] << ", ";
            writeFile << x_incumbent[index] << ", ";
        }
        std::cout << x_incumbent[A_colsize - 1] << std::endl;
        writeFile << x_incumbent[A_colsize - 1] << std::endl;
        std::cout << "k (number of nearest neighbor): " << k_new << std::endl;
        writeFile << "k (number of nearest neighbor): " << k_new << std::endl;
        k = k_new;
        old_knn_radius = knn_radius;
        // update candidates and incuments
        old_candidate = x_candidate;
        old_incumbent = x_incumbent;
        // output solution at the iteration pointer
        // syntax: iteration time dual_count minorant_count solution
        if (idx_pointer < it_pointer.size() && iteration == it_pointer[idx_pointer] - 1) {
            double duration2 = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
            writeFileSol << it_pointer[idx_pointer] << "," << duration2 << "," << explored_duals.size() << "," << minorant_collection.size();
            for (int idx_x = 0; idx_x < x_incumbent.size(); ++idx_x) {
                writeFileSol << "," << x_incumbent[idx_x];
            }
            writeFileSol << std::endl;
            idx_pointer++;
        }
    } // end main iteration
    std::cout << "*******************************************\n";
    writeFile << "*******************************************\n";
    std::cout << "Output Solution: ";
    writeFile << "Output Solution: ";
    for (int index = 0; index < A_colsize-1; ++index) {
        std::cout << x_incumbent[index] << ", ";
        writeFile << x_incumbent[index] << ", ";
    }
    std::cout << x_incumbent[A_colsize - 1] << std::endl;
    writeFile << x_incumbent[A_colsize - 1] << std::endl;
    std::cout << "Computation Log: Finish Solving Process.\n";
    writeFile << "Computation Log: Finish Solving Process.\n";
    // write time elapsed
    double duration = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    writeFile << "Time elapsed(secs) : " << duration << "\n";
    writeFile << "*******************************************\n";
    
    writeFile.close();
    writeFileSol.close();
    env_sub.end();
    // return the results
    solverOutput res;
    res.x = x_incumbent;
    res.num_it = max_iterations;
    res.time_elapse = duration;
    // end cplex environment
    return res;
} // end nsd_batch_solver

