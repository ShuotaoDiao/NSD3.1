//
//  OBD_utils.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 11/1/22.
//

#include "OBD_utils.hpp"

void interface_obd(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                  int N_pre,
                  int it_max,
                  double error) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    // obtain estimated solution from the OBD solver
    bd_output res_obd = obd_solver(folder_path, observed_predictor, N_pre, it_max, error);
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    // validate the solution quality
    validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, res_obd.x);
    int sample_size = N_pre + it_max;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/obd1.0_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    writeFile << it_max << ", ";
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << res_obd.max_gap << ", ";
    writeFile << res_obd.it_num << ", ";
    writeFile << res_obd.sol_flag << ", ";
    writeFile << res_obd.cuts_count << ", ";
    writeFile << time_elapse << std::endl;
}

void interface_obd2(const std::string& folder_path,
                  const std::string& validation_folder_path,
                  const std::vector<double>& observed_predictor,
                  int N_pre,
                  int it_max,
                  const std::vector<int>& it_pointer,
                    double error) {
    std::string outputResults_path = folder_path + "/obd1.1_summary2.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    bd_output res = obd_solver2(folder_path, observed_predictor, N_pre, it_max, it_pointer, error);
    // read file
    std::string readPath = folder_path + "/sol(obd_v1.1).txt";
    const char* readPathConst = readPath.c_str(); // convert the string type path to constant
    std::ifstream readFile(readPathConst); // create a readFile object
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile, line1)) { // get the whole line
            std::stringstream ss1(line1); // convert a string into stream
            unsigned int index_position = 0; // 1 iteration, 2 time, 3 total_it, 4 max_gap, 5 num_cuts, 6 solution
            std::vector<double> candidate_sol;
            while (getline(ss1, line1, ',')) {
                index_position += 1;
                std::stringstream ss2(line1);
                if (index_position == 1) { // iteration
                    int it;
                    ss2 >> it;
                    writeFile << it << ", ";
                    // sample
                    int total_sample = N_pre + it;
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
                else if (index_position == 5) {
                    int cut_count;
                    ss2 >> cut_count;
                    writeFile << cut_count << ", ";
                }
                else if (index_position > 5) {
                    double val;
                    ss2 >> val;
                    candidate_sol.push_back(val);
                }
            } // end while (getline(ss1, line1, ','))
            // validate the solution quality
            validationResult res_val = twoStageLP_validation_outputResults(validation_folder_path, candidate_sol);
            writeFile << res_val.mean << std::endl;
        } // end while (getline(readFile, line1))
    } // end if (readFile.is_open())
    writeFile.close();
}
