//
//  NSD_utils.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include "NSD_utils.hpp"

validationResult twoStageLP_validation_outputResults(const std::string& folder_path, const std::vector<double>& x_candidate) {
    // STEP 1: INITIALIZATION
    bool flag_be;
    bool flag_bi;
    bool flag_Ce;
    bool flag_Ci;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    const char* model_path_const = model_path.c_str();
    const char* sto_path_const = sto_path.c_str();
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
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
    RHSmap = readStochasticMap(sto_path);
    long sample_size = 0;
    if (flag_be == true) {
        sample_size = be_DB[0].size();
    }
    else if (flag_bi == true) {
        sample_size = bi_DB[0].size();
    }
    else if (flag_Ce == true) {
        sample_size = Ce_DB[0].size();
    }
    else if (flag_Ci == true) {
        sample_size = Ci_DB[0].size();
    }
    else {
        throw std::invalid_argument("ERROR: Database is empty!\n");
    }
    // determine appropriate model
    double secondStageTotalCost = 0;
    double firstStageCost = 0;
    for (int idx = 0; idx < model_parameters.c.getNzeroLen(); ++idx) {
        firstStageCost += model_parameters.c.getVal(idx) * x_candidate[model_parameters.c.getLoc(idx)];
    }
    double variance_P = 0; // intermediate component for calculating variance
    // initialize subproblem
    dataPoint be_datapoint;
    if (flag_be == true) {
        be_datapoint = be_DB[0][0];
    }
    dataPoint bi_datapoint;
    if (flag_bi == true) {
        bi_datapoint = bi_DB[0][0];
    }
    dataPoint Ce_datapoint;
    if (flag_Ce == true) {
        Ce_datapoint = Ce_DB[0][0];
    }
    dataPoint Ci_datapoint;
    if (flag_Ci == true) {
        Ci_datapoint = Ci_DB[0][0];
    }
    // merge all the datapoints
    secondStageRHSpoint RHS_cur = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
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
            exprs_eq_sub[model_parameters.C.getRow(idx)] += model_parameters.C.getVal(idx) * x_candidate[col_idx];
        }
    }
    // right hand side e
    for (int idx = 0; idx < model_parameters.e.getNzeroLen(); ++idx) {
        exprs_eq_sub[model_parameters.e.getLoc(idx)] -= model_parameters.e.getVal(idx);
    }
    // coefficients before x (stochastic part) equality
    for (int idx = 0; idx < RHS_cur.Ce.size(); ++idx) {
        exprs_eq_sub[RHSmap.Ce_map[idx].first] += RHS_cur.Ce[idx] * x_candidate[RHSmap.Ce_map[idx].second];
    }
    // coefficients before x (stochastic part) inequality (location is behind equality constraints)
    for (int idx = 0; idx < RHS_cur.Ci.size(); ++idx) {
        exprs_eq_sub[RHSmap.Ci_map[idx].first + model_parameters.num_eq] += RHS_cur.Ci[idx] * x_candidate[RHSmap.Ci_map[idx].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx = 0; idx < RHS_cur.be.size(); ++idx) {
        exprs_eq_sub[RHSmap.be_map[idx]] -= RHS_cur.be[idx];
    }
    // right hand side (stochastic part) equality bi_(i) inequality
    for (int idx = 0; idx < RHS_cur.bi.size(); ++idx) {
        exprs_eq_sub[RHSmap.bi_map[idx] + model_parameters.num_eq] -= RHS_cur.bi[idx];
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
    std::vector<double> rhs_bounds(model_parameters.D.getRowLength(), 0.0);
    std::vector<double> det_rhs_bounds = e_det;
    // det C x
    for (int col_idx = 0; col_idx < model_parameters.C.getColLength(); ++col_idx) {
        int beg_idx = model_parameters.C.getCbeg(col_idx);
        for (int idx = beg_idx; idx < model_parameters.C.getClen(col_idx) + beg_idx; ++idx) {
            det_rhs_bounds[model_parameters.C.getRow(idx)] -= model_parameters.C.getVal(idx) * x_candidate[col_idx];
        }
    }
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_scenario];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_scenario];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_scenario];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_scenario];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        // update the subproblem
        // compute a new dual at x_candidate
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
        double tempCost = cplex_sub.getObjValue();
        secondStageTotalCost += tempCost / ((double) sample_size);
        double tempTotalCost = tempCost + firstStageCost;
        double _n_ = idx_scenario + 1;
        variance_P = variance_P * (_n_ - 1) / _n_ + tempTotalCost * tempTotalCost / _n_;
    } // end for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario)
    double _n_ = sample_size;
    validationResult result;
    result.mean = firstStageCost + secondStageTotalCost;
    result.variance = (variance_P - result.mean * result.mean) * _n_ / (_n_ - 1);
    result.num_dataPoint = sample_size;
    double halfMargin = result.Zalpha * sqrt(result.variance / _n_);
    result.CI_lower = result.mean - halfMargin;
    result.CI_upper = result.mean + halfMargin;
    // write computational results
    std::string outputResults_path = folder_path + "/validationResults_NSD3.1.txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    writeFile << "***************************************************\n";
    writeFile << "Estimating the quality of candidate solution\n";
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    writeFile << "Candidate solution             : ";
    long x_size = x_candidate.size();
    for (int index = 0; index < x_size - 1; ++index) {
        writeFile << x_candidate[index] << ", ";
    }
    writeFile << x_candidate[x_size - 1] << "\n";
    writeFile << "Number of data points          : " << result.num_dataPoint << "\n";
    std::cout << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Variance                       : " << result.variance << "\n";
    writeFile << "Variance in estimating the mean: " << sqrt(result.variance/ _n_) << "\n";
    writeFile << result.alpha << "% confidence interval of expected cost: [" << result.CI_lower << ", " << result.CI_upper << "]\n";
    writeFile << "***************************************************\n";
    // end cplex environment
    env_sub.end();
    return result;
}


void interface_nsd(const std::string& folder_path,
                     const std::string& validation_folder_path,
                     int max_iterations,
                     double f_upperbound,
                     double f_lowerbound,
                     double sigma_upperbound,
                     double sigma_lowerbound,
                     const std::vector<double>& observed_predictor,
                     int N_pre,
                     bool flag_debug) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    // obtain estimated solution from NSD solver (fast)
    solverOutput res_nsd = nsd_solver(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //std::clock_t time_end = std::clock();
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    //double time_elapse = ((double) (std::clock() - time_start)) / CLOCKS_PER_SEC;
    //double time_elapse = (time_end - time_start) / (double) CLOCKS_PER_SEC;
    validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, res_nsd.x);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/nsd3.1_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    int sample_size = N_pre + max_iterations;
    writeFile << max_iterations << ", ";
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << time_elapse << std::endl;
}


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
                    bool flag_debug) {
    std::string outputResults_path = folder_path + "/sdknn3.1_summary2.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    solverOutput res = nsd_solver2(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, it_pointer,N_pre, flag_debug);
    // read file
    std::string readPath = folder_path + "/sol(sdknn_v3.1).txt";
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            unsigned int index_position = 0; // 1 iteration, 2 time, 3 dual_count, 4 minorant_count, 5 solution
            std::vector<double> candidate_sol;
            while (getline(ss1, line1, ',')) {
                index_position += 1;
                std::stringstream ss2(line1);
                if (index_position == 1) { // iteration
                    int it;
                    ss2 >> it;
                    writeFile << it << ", "; // iteration
                    // sample
                    int total_sample = N_pre + it;
                    writeFile << total_sample << ", "; // sample
                }
                else if (index_position == 2) {
                    double time_elapse;
                    ss2 >> time_elapse;
                    writeFile << time_elapse << ", ";
                }
                else if (index_position == 3) {
                    int dual_count;
                    ss2 >> dual_count;
                    writeFile << dual_count << ", ";
                }
                else if (index_position == 4) {
                    int minorant_count;
                    ss2 >> minorant_count;
                    writeFile << minorant_count << ", ";
                }
                else if (index_position > 4) {
                    double val;
                    ss2 >> val;
                    candidate_sol.push_back(val);
                }
            } // end while (getline(ss1, line1, ','))
            // validate the solution quality
            validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, candidate_sol);
            writeFile << res_val.mean << std::endl;;
        } // end while (getline(readFile, line1))
    }
    writeFile.close();
}


/*
 Output format: Iteration, Sample, Time, #Duals, #Minorants, Obj
 */
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
                         bool flag_debug) {
    std::string outputResults_path = folder_path + "/sdknn_batch3.1_summary2.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    solverOutput res = nsd_batch_solver(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, it_pointer, batch_init, batch_incre, flag_debug);
    // read file
    std::string readPath = folder_path + "/sol(sdknn_batch_v3.1).txt";
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            unsigned int index_position = 0; // 1 iteration, 2 time, 3 dual_count, 4 minorant_count, 5 solution
            std::vector<double> candidate_sol;
            while (getline(ss1, line1, ',')) {
                index_position += 1;
                std::stringstream ss2(line1);
                if (index_position == 1) { // iteration
                    int it;
                    ss2 >> it;
                    writeFile << it << ", "; // #iterations
                    // sample
                    int total_sample = batch_init;
                    int batch = batch_init;
                    for (int _it = 0; _it < it - 1; ++_it) {
                        batch += batch_incre;
                        total_sample += batch;
                    }
                    writeFile << total_sample << ", "; // sample
                }
                else if (index_position == 2) {
                    double time_elapse;
                    ss2 >> time_elapse;
                    writeFile << time_elapse << ", ";
                }
                else if (index_position == 3) {
                    int dual_count;
                    ss2 >> dual_count;
                    writeFile << dual_count << ", ";
                }
                else if (index_position == 4) {
                    int minorant_count;
                    ss2 >> minorant_count;
                    writeFile << minorant_count << ", ";
                }
                else if (index_position > 4) {
                    double val;
                    ss2 >> val;
                    candidate_sol.push_back(val);
                }
            } // end while (getline(ss1, line1, ','))
            // validate the solution quality
            validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, candidate_sol);
            writeFile << res_val.mean << std::endl;;
        } // end while (getline(readFile, line1))
    }
    writeFile.close();
}
