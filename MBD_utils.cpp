//
//  MBD_utils.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/30/22.
//

#include "MBD_utils.hpp"

std::vector<double> interface_mbd(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                   const std::vector<double>& x_init,
                  int sample_size,
                   double error) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    // obtain estimated solution from the OBD solver
    bd_output res_mbd = mbd_solver(folder_path, observed_predictor, x_init, sample_size, error);
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    // validate the solution quality
    validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, res_mbd.x);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/mbd1.1_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << res_mbd.max_gap << ", ";
    writeFile << res_mbd.it_num << ", ";
    writeFile << res_mbd.sol_flag << ", ";
    writeFile << time_elapse << std::endl;
    return res_mbd.x;
}


void interface_mbd2(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                   const std::vector<double>& x_init,
                  int sample_init,
                  int max_it,
                                   double error) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    bd_output res;
    res.x = x_init;
    int sample_size = sample_init;
    for (int it = 0; it < max_it; ++it) {
        sample_size++;
        // obtain estimated solution from the OBD solver
        bd_output res_mbd = mbd_solver(folder_path, observed_predictor, res.x, sample_size, error);
        res.x = res_mbd.x;
        res.sol_flag = res_mbd.sol_flag;
        res.it_num = res_mbd.it_num;
        res.max_gap = res_mbd.max_gap;
    }
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    // validate the solution quality
    validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, res.x);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/mbd1.1_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << res.max_gap << ", ";
    writeFile << res.it_num << ", ";
    writeFile << res.sol_flag << ", ";
    writeFile << time_elapse << std::endl;
}


void interface_ombd(const std::string& folder_path,
                    const std::string& validation_folder_path,
                    const std::vector<double>& observed_predictor,
                    const std::vector<double>& x_init,
                    int sample_init,
                    int max_it,
                    double error) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    bd_output res = ombd_solver(folder_path, observed_predictor, x_init, sample_init, max_it, error);
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    // validate the solution quality
    validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, res.x);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/ombd1.1_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    int sample_size = sample_init + max_it;
    writeFile << max_it << ", ";
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << res.max_gap << ", ";
    writeFile << res.mbd_count << ", ";
    writeFile << res.sol_flag << ", ";
    writeFile << time_elapse << std::endl;
}


void interface_ombd2(const std::string& folder_path,
                     const std::string& validation_folder_path,
                     const std::vector<double>& observed_predictor,
                     const std::vector<double>& x_init,
                     int sample_init,
                     int max_it,
                     const std::vector<int>& it_pointer,
                     double error) {
    std::string outputResults_path = folder_path + "/ombd1.1_summary2.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    bd_output res = ombd_solver2(folder_path, observed_predictor, x_init, sample_init, max_it, it_pointer, error);
    // read file
    std::string readPath = folder_path + "/sol(ombd_v1.1).txt";
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            unsigned int index_position = 0; // 1 iteration, 2 time, 3 total_it, 4 max_gap, 5 solution
            std::vector<double> candidate_sol;
            while (getline(ss1, line1, ',')) {
                index_position += 1;
                std::stringstream ss2(line1);
                if (index_position == 1) { // iteration
                    int it;
                    ss2 >> it;
                    writeFile << it << ", ";
                    // sample
                    int total_sample = sample_init + it;
                    writeFile << total_sample << ", ";
                }
                else if (index_position == 2) {
                    double time_elapse;
                    ss2 >> time_elapse;
                    writeFile << time_elapse << ", ";
                }
                else if (index_position == 3) {
                    int total_it;
                    ss2 >> total_it;
                    writeFile << total_it << ", ";
                }
                else if (index_position == 4) {
                    double max_gap;
                    ss2 >> max_gap;
                    writeFile << max_gap << ", ";
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
    } // end if (readFile.is_open())
    writeFile.close();
}


// OMBD with incremental batch sampling
void interface_ombd2_batch(const std::string& folder_path,
                           const std::string& validation_folder_path,
                           const std::vector<double>& observed_predictor,
                           const std::vector<double>& x_init,
                           int batch_init,
                           int batch_incre,
                           int max_it,
                           const std::vector<int>& it_pointer,
                           double error) {
    std::string outputResults_path = folder_path + "/ombd_batch1.1_summary2.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    bd_output res = ombd_solver2_batch(folder_path, observed_predictor, x_init, batch_init, batch_incre, max_it, it_pointer, error);
    // read file
    std::string readPath = folder_path + "/sol(ombd_batch_v1.1).txt";
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            unsigned int index_position = 0; // 1 iteration, 2 time, 3 total_it, 4 max_gap, 5 solution
            std::vector<double> candidate_sol;
            while (getline(ss1, line1, ',')) {
                index_position += 1;
                std::stringstream ss2(line1);
                if (index_position == 1) { // iteration
                    int it;
                    ss2 >> it;
                    writeFile << it << ", ";
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
                    int total_it;
                    ss2 >> total_it;
                    writeFile << total_it << ", ";
                }
                else if (index_position == 4) {
                    double max_gap;
                    ss2 >> max_gap;
                    writeFile << max_gap << ", ";
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
    } // end if (readFile.is_open())
    writeFile.close();
}
