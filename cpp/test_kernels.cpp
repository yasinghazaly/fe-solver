#include "kernels.hpp"
#include "iostream"


int main() {
    
    std::vector<std::int64_t>    indptr  = {0, 1, 3, 5};
    std::vector<std::int64_t>    indices = {0, 0, 1, 1, 2};
    std::vector<double> data    = {2.0, 1.0, 3.0, 4.0, 5.0};
    std::vector<double> r       = {4.0, 5.0, 14.0};
    std::vector<double> x = {1.0,2.0,3.0}; 
    std::int64_t n_fs = static_cast<int>(r.size()); 
    std::vector<double> y_fs(n_fs,0.0); 
    forward_solve(indptr.data(), indices.data(), data.data(),r.data(), n_fs, y_fs.data()); 
    for (double v : y_fs) std::cout << v << " ";
    std::cout << "\n"; 

    std::int64_t n_matvec = static_cast<int>(y_fs.size()); 
    std::vector<double> y_mv(n_matvec,0.0); 
    matvec(indptr.data(), indices.data(), data.data(),x.data(), n_matvec, y_mv.data());
    for (double v : y_mv) std::cout << v << " ";
    std::cout << "\n";   

    std::vector<std::int64_t>    indptr_t  = {0, 2, 4, 5};
    std::vector<std::int64_t>    indices_t = {0, 1, 1, 2, 2};
    std::vector<double> data_t   = {2.0, 1.0, 3.0, 4.0, 5.0};
    std::int64_t n_bs = static_cast<int>(r.size()); 
    std::vector<double> z_bs(n_bs,0.0); 
    backward_solve(indptr_t.data(), indices_t.data(), data_t.data(),y_fs.data(), n_bs, z_bs.data());
    for (double v : z_bs) std::cout << v << " ";
    std::cout << "\n"; 

    return 0;
}