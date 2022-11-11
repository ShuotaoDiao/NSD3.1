//
//  SparseVector.hpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#ifndef SparseVector_hpp
#define SparseVector_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

// Declaration of the class SparseVector
class SparseVector {
private:
    const double precision = 1e-9; // precision for nonzero values
    long len = 0; // vector length
    long nzero_len = 0; // number of nonzeros
    std::vector<int> ind; // index of the nonzero values in the vector
    std::vector<double> val; // nonzero values in the vector
public:
    SparseVector();
    SparseVector(std::vector<double> denseVec);
    SparseVector(const SparseVector& sv); // copy constructor
    void operator=(const SparseVector& sv);
    void setSparseVector(std::vector<double> denseVec);
    std::vector<double> scatter(); // convert sparse vector into dense vector (return a dense vector)
    void scatter(std::vector<double>& w); // convert sparse vector into dense vector
    void eraseWorkVector(std::vector<double>& w); // convert all the entries of the work vector into 0
    void display(); // print out the dense vector
    // insert
    void insert(int idx, double value);
    // set vector length
    void setLen(long new_len);
    // fast self addition
    void fast_add(double multiplier, const SparseVector& sv, std::vector<double>& w); // assume w is a dense zero vector
    // fast dot product
    double fast_dotProduct(const SparseVector& sv, std::vector<double>& w); // assume w is a dense zero vector
    double fast_dotProduct(const std::vector<double>& w);
    // negate
    void negate();
    // *******************************************
    // Functions for traversal
    // get value
    double getVal(int idx) const;
    // get nonzero len
    long getNzeroLen() const;
    // get vector len
    long getLen() const;
    // get location
    int getLoc(int idx) const;
    // *******************************************
    
};
#endif /* SparseVector_hpp */
