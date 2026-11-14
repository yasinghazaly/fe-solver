#include "kernels.hpp"


void forward_solve(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* r, const int n , double* y_fs) 
    {     
    for (int row = 0; row < n; ++row) { 
        int start = indptr[row]; 
        int end = indptr[row+1]; 
        double row_sum = 0.0; 

        for (int position = start ; position < end-1; ++position){ 
            row_sum += data[position] * y_fs[indices[position]]; 
        }
        y_fs[row] = ((r[row] - row_sum) / data[end-1]) ; 
    }
}

void backward_solve(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* y, const int n , double* z_bs) 
    {    
    for (int row = n-1; row >=0; --row) { 
        int start = indptr[row]; 
        int end = indptr[row+1]; 
        double row_sum = 0.0; 

        for (int position = start+1 ; position < end; ++position){ 
            row_sum += data[position] * z_bs[indices[position]]; 
        }
        z_bs[row] = ((y[row] - row_sum) / data[start]) ; 
    }
}

void matvec(const std::int64_t* indptr, const std::int64_t* indices, 
    const double* data, const double* x, const int n , double* y_mv) 
    {
        for (int row = 0; row < n; ++row) { 
            int start = indptr[row];
            int end = indptr[row+1]; 
            double row_sum = 0.0; 
            for (int position = start ; position <end; ++position){
                row_sum += data[position] * x[indices[position]]; 
            }
            y_mv[row] = row_sum; 
        }
    }