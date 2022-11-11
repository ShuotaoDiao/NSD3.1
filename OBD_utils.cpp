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
