#pragma once
#include <vector>
#include <cstdint>

void forward_solve(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* r, const int n , double* y_fs) ; 

void backward_solve(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* y_fs, const int n , double* z_bs) ; 

void matvec(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* x, const int n , double* y_mv) ; 
