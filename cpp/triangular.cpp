#include <iostream>
#include <vector>

std::vector<double> forward_solve(const std::vector<int>& indptr, const std::vector<int>& indices, 
    const std::vector<double>& data, const std::vector<double>& r) 
    {    
    int n = static_cast<int>(r.size()); 
    std::vector<double> y(n,0.0); 
    for (int row = 0; row < n; ++row) { 
        int start = indptr[row]; 
        int end = indptr[row+1]; 
        double row_sum = 0.0; 

        for (int position = start ; position < end-1; ++position){ 
            row_sum += data[position] * y[indices[position]]; 
        }
        y[row] = ((r[row] - row_sum) / data[end-1]) ; 
    }
    return y; 
}

std::vector<double> backward_solve(const std::vector<int>& indptr, const std::vector<int>& indices, 
    const std::vector<double>& data, const std::vector<double>& y) 
    {    
    int n = static_cast<int>(y.size()); 
    std::vector<double> z(n,0.0); 
    for (int row = n-1; row >=0; --row) { 
        int start = indptr[row]; 
        int end = indptr[row+1]; 
        double row_sum = 0.0; 

        for (int position = start+1 ; position < end; ++position){ 
            row_sum += data[position] * z[indices[position]]; 
        }
        z[row] = ((y[row] - row_sum) / data[start]) ; 
    }
    return z; 
}

std::vector<double> matvec(const std::vector<int>& indptr, const std::vector<int>& indices, 
    const std::vector<double>& data, const std::vector<double>& x) 
    {
        int n = static_cast<int>(x.size()); 
        std::vector<double> y(n,0.0); 
        for (int row = 0; row < n; ++row) { 
            int start = indptr[row];
            int end = indptr[row+1]; 
            double row_sum = 0.0; 
            for (int position = start ; position <end; ++position){
                row_sum += data[position] * x[indices[position]]; 
            }
            y[row] = row_sum; 
        }
        return y;
    }
 

int main() {
    
    std::vector<int>    indptr  = {0, 1, 3, 5};
    std::vector<int>    indices = {0, 0, 1, 1, 2};
    std::vector<double> data    = {2.0, 1.0, 3.0, 4.0, 5.0};
    std::vector<double> r       = {4.0, 5.0, 14.0};
    std::vector<double> x = {1.0,2.0,3.0}; 
    std::vector<double> y_fs = forward_solve(indptr, indices, data, r); 
    for (double v : y_fs) std::cout << v << " ";
    std::cout << "\n"; 

    std::vector<double> y_mv = matvec(indptr, indices, data, x);
    for (double v : y_mv) std::cout << v << " ";
    std::cout << "\n";   

    std::vector<int>    indptr_t  = {0, 2, 4, 5};
    std::vector<int>    indices_t = {0, 1, 1, 2, 2};
    std::vector<double> data_t   = {2.0, 1.0, 3.0, 4.0, 5.0};
    std::vector<double> z_bs = backward_solve(indptr_t, indices_t, data_t, y_fs);
    for (double v : z_bs) std::cout << v << " ";
    std::cout << "\n"; 

    return 0;
}