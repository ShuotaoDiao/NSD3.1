//
//  NSD_ioModel.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include "NSD_ioModel.hpp"

std::vector<std::vector<double>> standard_D(const std::vector<std::vector<double>>& De, const std::vector<std::vector<double>>& Di){
    // size of D
    //long n_row = De.size() + Di.size();
    long n_col = 0;
    long n_slack = 0;
    if (De.size() > 0) {
        n_col = De[0].size();
    }
    else if (Di.size() > 0) {
        n_col = Di[0].size();
    }
    if (Di.size() > 0) {
        n_col += Di.size(); // number of columns for slack variable
        n_slack = Di.size(); // equal to number of rows in the inequality constraints
    }
    // construct D
    std::vector<std::vector<double>> D;
    // block for equality
    for (int row = 0 ; row < De.size(); ++row) {
        std::vector<double> D_row = De[row];
        // add 0 matrix
        for (int idx = 0; idx < Di.size(); ++idx) {
            D_row.push_back(0.0);
        }
        D.push_back(D_row);
    }
    // identity block matrix for inequality
    for (int row = 0; row < Di.size(); ++row) {
        std::vector<double> D_row = Di[row];
        for (int col_slack = 0; col_slack < n_slack; ++col_slack) {
            if (row == col_slack) {
                D_row.push_back(1.0);
            }
            else {
                D_row.push_back(0.0);
            }
        }
        D.push_back(D_row);
    }
    return D;
}

std::vector<std::vector<double>> standard_C(const std::vector<std::vector<double>>& Ce,
                                            const std::vector<std::vector<double>>& Ci) {
    std::vector<std::vector<double>> C;
    for (int row = 0; row < Ce.size(); ++row) {
        std::vector<double> C_row = Ce[row];
        C.push_back(C_row);
    }
    for (int row = 0; row < Ci.size(); ++row) {
        std::vector<double> C_row = Ci[row];
        C.push_back(C_row);
    }
    return C;
}

std::vector<double> standard_e(const std::vector<double>& be, const std::vector<double>& bi) {
    std::vector<double> e;
    for (int idx = 0; idx < be.size(); ++idx) {
        e.push_back(be[idx]);
    }
    for (int idx = 0; idx < bi.size(); ++idx) {
        e.push_back(bi[idx]);
    }
    return e;
}

// standard model parameters
standardTwoStageParameters readStandardTwoStageParameters(const std::string& parameterPath) {
    // temporary variables (non-sparse)
    std::vector<double> c;
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    std::vector<double> d;
    std::vector<std::vector<double>> Di;
    std::vector<std::vector<double>> Ci;
    std::vector<std::vector<double>> De;
    std::vector<std::vector<double>> Ce;
    std::vector<double> bi;
    std::vector<double> be;
    const std::string nameBeginModel("<model:twoStageLP>");
    const std::string nameEndModel("</model:twoStageLP>");
    const std::string nameBeginParameter_c("<c>");
    const std::string nameEndParameter_c("</c>");
    const std::string nameBeginParameter_A("<A>");
    const std::string nameEndParameter_A("</A>");
    const std::string nameBeginParameter_b("<b>");
    const std::string nameEndParameter_b("</b>");
    const std::string nameBeginParameter_d("<d>");
    const std::string nameEndParameter_d("</d>");
    const std::string nameBeginParameter_Di("<Di>");
    const std::string nameEndParameter_Di("</Di>");
    const std::string nameBeginParameter_Ci("<Ci>");
    const std::string nameEndParameter_Ci("</Ci>");
    const std::string nameBeginParameter_De("<De>");
    const std::string nameEndParameter_De("</De>");
    const std::string nameBeginParameter_Ce("<Ce>");
    const std::string nameEndParameter_Ce("</Ce>");
    const std::string nameBeginParameter_bi("<bi>");
    const std::string nameEndParameter_bi("</bi>");
    const std::string nameBeginParameter_be("<be>");
    const std::string nameEndParameter_be("</be>");
    std::string readCondition("null");
    const char* parameterPathConst = parameterPath.c_str(); // convert a string path into a constant path
    std::ifstream readFile(parameterPathConst);
    if (readFile.is_open()) {
        std::string line1;
        while (getline(readFile,line1)) {// get the whole line
            //std::cout << line1 << std::endl; // for debug
            std::stringstream ss1(line1); // convert a string into stream
            if (nameBeginModel.compare(line1) != 0 && nameEndModel.compare(line1) != 0) { // main content
                if (nameBeginParameter_c.compare(line1) == 0) { // beign reading parameter c
                    readCondition = "c";
                }
                else if (nameBeginParameter_A.compare(line1) == 0) { // begin reading parameter A
                    readCondition = "A";
                }
                else if (nameBeginParameter_b.compare(line1) == 0) { // begin reading parameter b
                    readCondition = "b";
                }
                else if (nameBeginParameter_d.compare(line1) == 0) { // begin reading parameter d
                    readCondition = "d";
                }
                else if (nameBeginParameter_Di.compare(line1) == 0) { // begin reading parameter Di
                    readCondition = "Di";
                }
                else if (nameBeginParameter_Ci.compare(line1) == 0) { // begin reading parameter Ci
                    readCondition = "Ci";
                }
                else if (nameBeginParameter_De.compare(line1) == 0) { // begin reading parameter De
                    readCondition = "De";
                }
                else if (nameBeginParameter_Ce.compare(line1) == 0) { // begin reading parameter Ce
                    readCondition = "Ce";
                }
                else if (nameBeginParameter_bi.compare(line1) == 0) { // begin reading parameter bi
                    readCondition = "bi";
                }
                else if (nameBeginParameter_be.compare(line1) == 0) { // begin reading be
                    readCondition = "be";
                }
                else if (nameEndParameter_c.compare(line1) == 0) { // end reading parameter c
                    readCondition = "null";
                }
                else if (nameEndParameter_A.compare(line1) == 0) { // end reading parameter A
                    readCondition = "null";
                }
                else if (nameEndParameter_b.compare(line1) == 0) { // end reading parameter b
                    readCondition = "null";
                }
                else if (nameEndParameter_d.compare(line1) == 0) { // end reading parameter d
                    readCondition = "null";
                }
                else if (nameEndParameter_Di.compare(line1) == 0) { // end reading parameter Di
                    readCondition = "null";
                }
                else if (nameEndParameter_Ci.compare(line1) == 0) { // end reading parameter Ci
                    readCondition = "null";
                }
                else if (nameEndParameter_De.compare(line1) == 0) { // end reading parameter De
                    readCondition = "null";
                }
                else if (nameEndParameter_Ce.compare(line1) == 0) { // end reading parameter Ce
                    readCondition = "null";
                }
                else if (nameEndParameter_bi.compare(line1) == 0) { // end reading parameter bi
                    readCondition = "null";
                }
                else if (nameEndParameter_be.compare(line1) == 0) { // end reading parameter be
                    readCondition = "null";
                }
                else {
                    if (readCondition.compare("c") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                c.push_back(value);
                            }
                        }
                    }
                    else if (readCondition.compare("A") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            std::vector<double> A_row; // create one new row of A matrix
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                A_row.push_back(value);
                            }
                            A.push_back(A_row);
                        }
                    }
                    else if (readCondition.compare("b") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                b.push_back(value);
                            }
                        }
                    }
                    else if (readCondition.compare("d") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                d.push_back(value);
                            }
                        }
                    }
                    else if (readCondition.compare("Di") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            std::vector<double> Di_row; // create one new row of Di matrix
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                Di_row.push_back(value);
                            }
                            Di.push_back(Di_row);
                        }
                    }
                    else if (readCondition.compare("Ci") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            std::vector<double> Ci_row; // create one new row of Ci matrix
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                Ci_row.push_back(value);
                            }
                            Ci.push_back(Ci_row);
                        }
                    }
                    else if (readCondition.compare("De") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            std::vector<double> De_row; // create one new row of De matrix
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                De_row.push_back(value);
                            }
                            De.push_back(De_row);
                        }
                    }
                    else if (readCondition.compare("Ce") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            std::vector<double> Ce_row; // create one new row of Ce matrix
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                Ce_row.push_back(value);
                            }
                            Ce.push_back(Ce_row);
                        }
                    }
                    else if (readCondition.compare("bi") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                bi.push_back(value);
                            }
                        }
                    }
                    else if (readCondition.compare("be") == 0) {
                        while (getline(ss1, line1, ';')) {
                            std::stringstream ss2(line1);
                            while (getline(ss2, line1, ',')) {
                                double value;
                                std::stringstream ss3(line1);
                                ss3 >> value;
                                be.push_back(value);
                            }
                        }
                    }
                }
            }
            //std::cout << "Condition : " << readCondition << std::endl; // for debug
        }
    }
    readFile.close();
    // initialize variables
    standardTwoStageParameters parameters;
    // store the number of inequalities and number of equalities
    parameters.num_eq = De.size();
    parameters.num_ineq = Di.size();
    // convert to standard form
    std::vector<std::vector<double>> D = standard_D(De, Di);
    std::vector<std::vector<double>> C = standard_C(Ce, Ci);
    std::vector<double> e = standard_e(be, bi);
    parameters.c.setSparseVector(c);
    parameters.b.setSparseVector(b);
    parameters.A.setSparseMatrix(A);
    parameters.C.setSparseMatrix(C);
    parameters.D.setSparseMatrix(D);
    parameters.e.setSparseVector(e);
    parameters.d.setSparseVector(d);
    return parameters;
}
