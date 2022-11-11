//
//  main.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include <iostream>

#include "NSD_utils.hpp"
#include "OBD_utils.hpp"
// new bk19
void nsd_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    for (int idx = 12; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}

void obd_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    //bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    double error = 1e-3;
    for (int idx = 12; idx < 13; ++idx) {
        interface_obd(folder_path, validation_path, observed_predictor, N_pre, max_iterations[idx], error);
    }
}

// bk19_3
void nsd_bk19_3() {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_3/experiment6/case1";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}

void nsd_bk19_3(int caseNumber) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_3/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}

void obd_bk19_3(int caseNumber) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_3/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    //bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    double error = 1e-3;
    for (int idx = 0; idx < 13; ++idx) {
        interface_obd(folder_path, validation_path, observed_predictor, N_pre, max_iterations[idx], error);
    }
}

void nsd_bk19_5() {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_5/experiment6/case1";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(5,0.0);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    observed_predictor[3] = 0.5;
    observed_predictor[4] = 0.5;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    for (int idx = 12; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}


void nsd_bk19_5(int caseNumber) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_5/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(5,0.0);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    observed_predictor[3] = 0.5;
    observed_predictor[4] = 0.5;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}


void nsd_bk19_7(int caseNumber) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_7/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(7,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    //int N_pre = 30000;
    //int N_pre = 30000;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
}


// old bk19
void nsd_bk19() {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/case8";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/trueValidation";
    //int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -0.3626;//(-0.3626, 0.5871, -0.2987)
    observed_predictor[1] = 0.5871;
    observed_predictor[2] = -0.2987;
    double f_upperbound = 2000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
    
} // end nsd_bk19


void nsd_bk19(int caseNumber) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/trueValidation";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] ={1, 5};
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -0.3626;//(-0.3626, 0.5871, -0.2987)
    observed_predictor[1] = 0.5871;
    observed_predictor[2] = -0.2987;
    double f_upperbound = 2000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    for (int idx = 0; idx < 13; ++idx) { // do 0 ... 12
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
    
} // end nsd_bk19


void nsd_baa99_large() {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/case1";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    //int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {903, 4710};
    int max_iterations[] = {2445};
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    double f_upperbound = 10000;
    double f_lowerbound = -110000; // -100000, -20000(too high)
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    //int N_pre = 49;
    //int N_pre = 6000; // experiment 8
    int N_pre = 30000; // experiment 9
    for (int idx = 0; idx < 1; ++idx) { // do 0 1
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
} // end nsd_baa99_large

void nsd_baa99_large2(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    //int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {903, 4710}; // experiment 8
    int max_iterations[] = {2445}; // experiment 9
    //int max_iterations[] = {100};
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    double f_upperbound = 10000;
    double f_lowerbound = -110000; // -100000, -20000(too high)
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    //int N_pre = 49;
    //int N_pre = 6000; // experiment 8
    int N_pre = 30000; // experiment 9
    for (int idx = 0; idx < 1; ++idx) { // do 0 1
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
} // end nsd_baa99_large2


void obd_baa99(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    //int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {903, 4710}; // experiment 8
    int max_iterations[] = {2445}; // experiment 9
    //int max_iterations[] = {2445};
    //int max_iterations[] ={1, 5};
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    //bool flag_debug = false;
    //int N_pre = 49;
    //int N_pre = 6000; // experiment 8
    int N_pre = 30000; // experiment 9
    double error = 1e-3;
    for (int idx = 0; idx < 1; ++idx) {
        interface_obd(folder_path, validation_path, observed_predictor, N_pre, max_iterations[idx], error);
    }
}


int main(int argc, const char * argv[]) {
    //nsd_bk19_3();
    //nsd_bk19_5();
    int dimension = 13;
    for (int caseNumber = 1; caseNumber < 11; ++caseNumber) {
        //nsd_new_bk19(caseNumber, dimension);
        obd_new_bk19(caseNumber, dimension);
        //nsd_bk19_3(caseNumber);
        //obd_bk19_3(caseNumber);
        //nsd_baa99_large2(caseNumber);
        //obd_baa99(caseNumber);
    }
     
    /*
    for (int caseNumber = 2; caseNumber < 11; ++caseNumber) {
        //nsd_bk19_5(caseNumber);
        nsd_bk19_7(caseNumber);
    }
     */
    //nsd_bk19();
    /*
    for (int caseNumber = 6; caseNumber < 13; ++caseNumber) {
        nsd_bk19(caseNumber);
    }
     */
    /*
    for (int caseNumber = 1; caseNumber < 11; ++caseNumber) {
        nsd_baa99_large2(caseNumber);
    }
    */
    return 0;
}
