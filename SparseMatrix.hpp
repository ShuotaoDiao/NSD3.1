//
//  SparseMatrix.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef SparseMatrix_hpp
#define SparseMatrix_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include "SparseVector.hpp"

class SparseMatrix {
private:
    const double precision = 1e-9; // precision for nonzero values
    long row_length = 0;
    long col_length = 0;
    // store the matrix a collection of columns
    std::vector<int> cbeg; // beginning index of the nonzero entry
    std::vector<int> clen; // number of nonzero entries in each column
    std::vector<int> rind; // row index
    std::vector<double> val; // entry value
public:
    SparseMatrix(); // default constructor
    SparseMatrix(const std::vector<std::vector<double>>& denseMatrix);
    SparseMatrix(const SparseMatrix& sm); // copy constructor
    void operator=(const SparseMatrix& sm);
    void setSparseMatrix(const std::vector<std::vector<double>>& denseMatrix);
    // fast right multiply, assume that m is a zero matrix and w is a zero vector; x' * M
    SparseVector fast_rightMultiply(SparseVector& sv, std::vector<double>& w);
    std::vector<double> fast_rightMultiply(std::vector<double> v);
    // fast left multiply, assume that v and w are a zero vectors; M * x
    SparseVector fast_leftMultiply(SparseVector& sv, std::vector<double>& w, std::vector<double>& v);
    // *******************************************
    // Functions for traversal
    // get row
    int getRow(int idx) const;
    // get entry value
    double getVal(int idx) const;
    // get number of nonzero entries in each column
    int getClen(int idx) const;
    // get the beginning index
    int getCbeg(int idx) const;
    // get row length
    long getRowLength() const;
    // get column length
    long getColLength() const;
    // *******************************************
};
#endif /* SparseMatrix_hpp */
