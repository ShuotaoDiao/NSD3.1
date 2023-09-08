//
//  main.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include <iostream>

#include "NSD_utils.hpp"
#include "OBD_utils.hpp"
#include "MBD_utils.hpp"
// new bk19

void mbd_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    double error = 1e-6;
    //int sample_size = 8645;
    int sample_init = 49;
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    std::vector<double> x_init(4,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    for (int idx = 0; idx < 1; ++idx) {
        interface_mbd2(folder_path, validation_path, observed_predictor, x_init, sample_init, max_iterations[idx], error);
    }
    /*
    for (int idx = 0; idx < 1; ++idx) {
        x_init = interface_mbd(folder_path, validation_path, observed_predictor, x_init, sample_size, error);
    }
     */
}

void ombd_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    double error = 1e-6;
    //int sample_size = 8645;
    int sample_init = 49;
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    std::vector<double> x_init(4,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    for (int idx = 0; idx < 13; ++idx) {
        interface_ombd(folder_path, validation_path, observed_predictor, x_init, sample_init, max_iterations[idx], error);
    }
}


void ombd2_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    double error = 1e-6;
    //int sample_size = 8645;
    int sample_init = 49;
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    std::vector<double> x_init(4,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 13; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_ombd2(folder_path, validation_path, observed_predictor, x_init, sample_init, max_iterations[12], it_pointer, error);
}


void obd2_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    double error = 1e-6;
    //int sample_size = 8645;
    int sample_init = 49;
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    std::vector<double> x_init(4,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 13; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_obd2(folder_path, validation_path, observed_predictor, sample_init, max_iterations[12], it_pointer, error);
}


void ombd2_batch_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    double error = 1e-6;
    //int sample_size = 8645;
    int batch_init = 50;
    int batch_incre = 1;
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    std::vector<double> x_init(4,0.0);
    int max_iterations[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105}; // align with number of NSQG updates in the LEONA
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 14; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_ombd2_batch(folder_path, validation_path, observed_predictor, x_init, batch_init, batch_incre, max_iterations[12], it_pointer, error);
}



void nsd2_new_bk19(int caseNumber, int dimension) {
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
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 13; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_nsd2(folder_path, validation_path, max_iterations[12], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, it_pointer, N_pre, flag_debug);
}

// batch samlpig version of nsd
void nsd2_batch_new_bk19(int caseNumber, int dimension) {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19_";
    folder_path = folder_path + std::to_string(dimension) + "/experiment6/case" + std::to_string(caseNumber);
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/bk19/validationSet";
    int max_iterations[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105}; // align with number of NSQG updates in the LEONA
    std::vector<double> observed_predictor(dimension,0.5);
    observed_predictor[0] = -1.1401;//-1.1401, 0.3406, 1.3871
    observed_predictor[1] = 0.3406;
    observed_predictor[2] = 1.3871;
    double f_upperbound = 10000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int batch_init = 50;
    int batch_incre = 1;
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 14; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_nsd_batch(folder_path, validation_path, max_iterations[13], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, it_pointer, batch_init, batch_incre, flag_debug);
    
}

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
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {903, 4710};
    //int max_iterations[] = {2445};
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
    int N_pre = 49;
    //int N_pre = 6000; // experiment 8
    //int N_pre = 30000; // experiment 9
    for (int idx = 12; idx < 13; ++idx) { // do 0 1
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
} // end nsd_baa99_large

void nsd_baa99_large2(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    //int max_iterations[] = {903, 4710}; // experiment 8
    //int max_iterations[] = {2445}; // experiment 9
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
    int N_pre = 49;
    //int N_pre = 6000; // experiment 8
    //int N_pre = 30000; // experiment 9
    for (int idx = 11; idx < 12; ++idx) { // do 0 1
        interface_nsd(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
} // end nsd_baa99_large2


// batch sampling version of sdknn
void nsd_batch_baa99_large2(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    int max_iterations[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91}; // align with number of NSQG updates in the LEONA
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
    int batch_init = 50;
    int batch_incre = 1;
    //int N_pre = 6000; // experiment 8
    //int N_pre = 30000; // experiment 9
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 13; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_nsd_batch(folder_path, validation_path, max_iterations[11], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, it_pointer, batch_init, batch_incre, flag_debug);
}


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

void obd2_baa99(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    //bool flag_debug = false;
    int N_pre = 49;
    double error = 1e-6;
    std::vector<double> x_init(50,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 11; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_obd2(folder_path, validation_path, observed_predictor, N_pre, max_iterations[10], it_pointer, error);
}

void ombd_pbaa99(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    //bool flag_debug = false;
    int N_pre = 49;
    double error = 1e-6;
    std::vector<double> x_init(50,0.0);
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5396, 6854, 8596, 10661, 13091}; // 200
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 11; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_ombd2(folder_path, validation_path, observed_predictor, x_init, N_pre, max_iterations[10], it_pointer, error);
}


// PBAA99 OMBD with incremental batch sampling


void ombd_batch_pbaa99(int caseNumber) {
    //std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case" + std::to_string(caseNumber);
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/experiment2/case" + std::to_string(caseNumber);
    //std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment9/kNNValidation2";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/pbaa99/kNNValidation1";
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    //bool flag_debug = false;
    int batch_init = 50;
    int batch_incre = 1;
    double error = 1e-6;
    std::vector<double> x_init(50,0.0);
    int max_iterations[] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91}; // align with number of NSQG updates in the LEONA
    std::vector<int> it_pointer;
    for (int idx = 0; idx < 13; ++idx) {
        it_pointer.push_back(max_iterations[idx]);
    }
    interface_ombd2_batch(folder_path, validation_path, observed_predictor, x_init, batch_init, batch_incre, max_iterations[11], it_pointer, error);
}


int main(int argc, const char * argv[]) {
    //nsd_bk19_3();
    //nsd_bk19_5();
    /*
    int dimension = 5;
    // note: ombd2 dim = 13, case = 20
    for (int caseNumber = 20; caseNumber < 31; ++caseNumber) {
        //ombd2_new_bk19(caseNumber, dimension);
        nsd2_new_bk19(caseNumber, dimension);
        //nsd_new_bk19(caseNumber, dimension);
        //mbd_new_bk19(caseNumber, dimension);
        //ombd_new_bk19(caseNumber, dimension);
        //obd_new_bk19(caseNumber, dimension);
        //nsd_bk19_3(caseNumber);
        //obd_bk19_3(caseNumber);
        //nsd_baa99_large2(caseNumber);
        //obd_baa99(caseNumber);
        //ombd_pbaa99(caseNumber);
    }
     */
    
    for (int idx_dim = 6; idx_dim < 7; ++idx_dim) {
        int dimension = 3 + idx_dim * 2;
        for (int caseNumber = 1; caseNumber < 31; ++caseNumber) {
            //nsd2_new_bk19(caseNumber, dimension);
            //nsd2_batch_new_bk19(caseNumber, dimension);
            //ombd2_batch_new_bk19(caseNumber, dimension);
            obd2_new_bk19(caseNumber, dimension);
        }
    }
     
    //nsd2_batch_new_bk19(1, 3);
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
    for (int caseNumber = 30; caseNumber < 31; ++caseNumber) {
        //nsd_baa99_large2(caseNumber);
        //nsd_batch_baa99_large2(caseNumber);
        //ombd_batch_pbaa99(caseNumber);
        obd2_baa99(caseNumber);
    }
    */
    return 0;
}
