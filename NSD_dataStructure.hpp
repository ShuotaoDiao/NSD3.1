//
//  NSD_dataStructure.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef NSD_dataStructure_hpp
#define NSD_dataStructure_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <ilcplex/ilocplex.h>
#include <ctime>
#include <stdlib.h>
#include <cassert>
#include <unordered_map>
#include <map>
#include <utility>

#include "SparseMatrix.hpp"

// ****************************************************
// Target: NSD_solver
// dual multipliers
struct dualMultipliers {
    std::vector<double> dual; // equality constriant
    std::vector<double> sto_pi_e; // store temporary stochastic pi e values 
    double l1_norm;
    bool feasible_flag;
};
// feasibility cut $a^\top x \leq b$
struct feasibilityCut {
    SparseVector A_newRow;
    double b_newRow;
};

// ****************************************************
// Target: NSD_solver and NSD_ioModel
// standard two stage parameters
// A x <= b
// Dy = e - Cx
struct standardTwoStageParameters {
    // first stage c' x, A x <= b
    SparseVector c;
    SparseMatrix A;
    SparseVector b;
    // second stage Dy = e - Cx
    SparseVector d;
    SparseMatrix D;
    SparseMatrix C;
    SparseVector e;
    // extra paramters
    long num_eq = 0;
    long num_ineq = 0;
    double x_lb = 0.0;
    double x_ub = 1e6;
    double y_lb = 0.0;
};

// ****************************************************
// Target: ioDB
// data structures for the dataPoint
struct dataPoint { // definition of dataPoint
    std::vector<double> predictor;
    std::vector<double> response;
    double weight;
};

// ****************************************************
// Target:ioStochastic
// data structure
struct randomVector {
    std::vector<double> component; // all the entries of a random vector
    std::vector<int> randomIndices; // indices of random entries
};

struct randomScalar {
    double component = 0;
    bool flag_random = false; // flag which tells whether this scalar is random
};

// vectors on the right hand side of second stage problem
struct secondStageRHS {
    randomVector be;
    randomVector bi;
};
// database of vectors on the right hand side of second stage (new)
struct secondStageRHSDB {
    std::vector<std::vector<std::vector<double>>> be_database;
    std::vector<std::vector<std::vector<double>>> bi_database;
    std::vector<std::vector<std::vector<double>>> Ce_database;
    std::vector<std::vector<double>> weight_database;
};

struct secondStageRHSpoint {
    std::vector<double> be;
    std::vector<double> bi;
    std::vector<double> Ce;
    std::vector<double> Ci;
    std::vector<double> predictor;
};

// store the location of randomness
struct secondStageRHSmap {
    std::vector<int> be_map;
    std::vector<int> bi_map;
    std::vector<std::pair<int,int>> Ce_map;
    std::vector<std::pair<int,int>> Ci_map;
};

// ****************************************************
// Target: NSD_solver
struct validationResult {
    double mean;
    double variance;
    const double alpha = 95;
    const double Zalpha = 1.96;
    double CI_lower;
    double CI_upper;
    int num_dataPoint;
};


// ****************************************************
// Target: NSD_solver
// minorant
struct minorant {
    double alpha = 0;
    std::vector<double> beta;
    bool if_active = true;
};

// solver information
struct solverOutput {
    std::vector<double> x;
    int num_it = 0;
    double time_elapse = 0;
};


// ****************************************************
// Target: BD_solver
struct bd_solution {
    std::vector<double> x;
    std::vector<double> z;
    unsigned int sol_flag = 0; // 0 optimal solution exists; 1 infeasible or unbounded
};

struct bd_subproblem_cuts {
    std::vector<double> alpha_array;
    std::vector<std::vector<double>> beta_array;
};

struct bd_output {
    std::vector<double> x;
    double max_gap;
    unsigned int sol_flag;
    int it_num;
    int cuts_count;
    int mbd_count;
};

// functions
double operator*(const std::vector<double>& vec1, const std::vector<double>& vec2);

std::vector<double> operator*(double a, const std::vector<double>& vec1);

std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2);

double max(double a, double b);

double min(double a, double b);
#endif /* NSD_dataStructure_hpp */
