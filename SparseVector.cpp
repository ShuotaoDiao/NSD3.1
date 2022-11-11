//
//  SparseVector.cpp
//  NSD3.1
//
//  Created by Shuotao Diao on 10/14/22.
//

#include "SparseVector.hpp"

// default constructor
SparseVector::SparseVector() {
    
}

SparseVector::SparseVector(std::vector<double> denseVec) {
    // length of the vector
    len = denseVec.size();
    nzero_len = 0;
    // store nonzero values and corresponding indices
    for (int idx = 0; idx < len; ++idx) {
        if (denseVec[idx] > precision || denseVec[idx] < -precision) {
            ind.push_back(idx);
            val.push_back(denseVec[idx]);
            nzero_len += 1;
        }
    }
}

// copy constructor
SparseVector::SparseVector(const SparseVector& sv) {
    len = sv.len;
    nzero_len = sv.nzero_len;
    ind = sv.ind;
    val = sv.val;
}

void SparseVector::operator=(const SparseVector& sv) {
    len = sv.len;
    nzero_len = sv.nzero_len;
    ind = sv.ind;
    val = sv.val;
}

// scatter: convert sparse vector into dense vector O(n) complexity
std::vector<double> SparseVector::scatter() {
    std::vector<double> denseVec(len,0);
    for (int idx = 0; idx < nzero_len; ++idx) {
        denseVec[ind[idx]] = val[idx];
    }
    return denseVec;
}

// set the sparse vector based on input vector
void SparseVector::setSparseVector(std::vector<double> denseVec) {
    // length of the vector
    len = denseVec.size();
    nzero_len = 0;
    // store nonzero values and corresponding indices
    for (int idx = 0; idx < len; ++idx) {
        if (denseVec[idx] > precision || denseVec[idx] < -precision) {
            ind.push_back(idx);
            val.push_back(denseVec[idx]);
            nzero_len += 1;
        }
    }
}

// scatter: convert sparse vector into dense vector
void SparseVector::scatter(std::vector<double>& w) {
    for (int idx = 0; idx < nzero_len; ++idx) {
        w[ind[idx]] = val[idx];
    }
}

// convert all the entries of the work vector into 0
void SparseVector::eraseWorkVector(std::vector<double>& w) {
    for (int idx = 0; idx < nzero_len; ++idx) {
        if (w[ind[idx]] > precision || w[ind[idx]] < -precision) {
            w[ind[idx]] = 0;
        }
    }
}

// print out the dense vector
void SparseVector::display() {
    std::cout << "Display the Sparse Vector:\n";
    std::cout << "Length: " << len << std::endl;
    std::cout << "Index, Value\n";
    for (int idx = 0; idx < nzero_len; ++idx) {
        std::cout << ind[idx] << ", " << val[idx] << std::endl;
    }
}

// insert (warning, it may affect the integrity of the sparse vector, only for internal computation)
void SparseVector::insert(int idx, double value) {
    if (value > precision || value < -precision) { // value is not zero
        ind.push_back(idx);
        val.push_back(value);
        nzero_len += 1;
    }
}

// set vector length (warning, it may affect the integrity of the sparse vector, only for internal computation)
void SparseVector::setLen(long new_len) {
    len = new_len;
}


// fast self addition, assume w is a dense zero vector
void SparseVector::fast_add(double multiplier, const SparseVector& sv, std::vector<double>& w) {
    // scatter sv into w
    for (int idx = 0; idx < sv.nzero_len; ++idx) {
        w[sv.ind[idx]] = sv.val[idx];
    }
    // scan the indices of nonzero entries of itself
    for (int idx = 0; idx < nzero_len; ++idx) {
        if (w[ind[idx]] > precision || w[ind[idx]] < -precision) { // w[ind[idx]] is not zero
            val[idx] += multiplier * w[ind[idx]];
            w[ind[idx]] = 0; // reset w[ind[idx]] to 0
        }
    }
    // scan the indices of the nonzero entires of sv
    for (int idx = 0; idx < sv.nzero_len; ++idx) {
        if (w[sv.ind[idx]] > precision || w[sv.ind[idx]] < -precision) { // fill in
            ind.push_back(sv.ind[idx]);
            val.push_back(multiplier * sv.val[idx]);
            nzero_len += 1;
            w[sv.ind[idx]] = 0; // reset w[sv.ind[idx]] to 0;
        }
    }
}
// fast dot product, assume w is a dense zero vector
double SparseVector::fast_dotProduct(const SparseVector& sv, std::vector<double>& w) {
    // scatter sv into w
    for (int idx = 0; idx < sv.nzero_len; ++idx) {
        w[sv.ind[idx]] = sv.val[idx];
    }
    double dotprod = 0;
    for (int idx = 0; idx < nzero_len; ++idx) {
        dotprod += val[idx] * w[ind[idx]];
    }
    // restore w to 0 vector
    for (int idx = 0; idx < sv.nzero_len; ++idx) {
        w[sv.ind[idx]] = 0;
    }
    return dotprod;
}

double SparseVector::fast_dotProduct(const std::vector<double>& w) {
    double dotprod = 0;
    for (int idx = 0; idx < nzero_len; ++idx) {
        dotprod += val[idx] * w[ind[idx]];
    }
    return dotprod;
}

// negate
void SparseVector::negate() {
    for (int idx = 0; idx < nzero_len; ++idx) {
        val[idx] = -val[idx];
    }
}


// get value
double SparseVector::getVal(int idx) const{
    if (idx >= nzero_len) {
        throw std::out_of_range("SparseVector::getVal(int idx) Error: Index is out of range.\n");
    }
    return val[idx];
}

// get nonzero len
long SparseVector::getNzeroLen() const{
    return nzero_len;
}

// get vector len
long SparseVector::getLen() const {
    return len;
}

// get location
int SparseVector::getLoc(int idx) const{
    if (idx >= nzero_len) {
        throw std::out_of_range("SparseVector::getLoc(int idx) Error: Index is out of range.\n");
    }
    return ind[idx];
}


