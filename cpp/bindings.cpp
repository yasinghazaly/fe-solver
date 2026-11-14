#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels.hpp"

namespace py = pybind11; 

py::array_t<double> forward_solve_py(const py::array_t<std::int64_t> indptr, 
    const py::array_t<std::int64_t> indices, const py::array_t<double> data, 
    const py::array_t<double> r )
    {
        std::int64_t n_fs = static_cast<std::int64_t>(r.size()); 
        py::array_t<double> y_fs(n_fs);
        forward_solve(indptr.data(), indices.data(), data.data(),r.data(), n_fs, y_fs.mutable_data()); 

        return y_fs; 
    }

py::array_t<double> backward_solve_py(const py::array_t<std::int64_t> indptr, 
    const py::array_t<std::int64_t> indices, const py::array_t<double> data, 
    const py::array_t<double> y_fs)
    {
        std::int64_t n_bs = static_cast<std::int64_t>(y_fs.size()); 
        py::array_t<double> z_bs(n_bs);
        backward_solve(indptr.data(), indices.data(), data.data(),y_fs.data(), n_bs, z_bs.mutable_data()); 

        return z_bs; 
    }

py::array_t<double> matvec_py(const py::array_t<std::int64_t> indptr, 
    const py::array_t<std::int64_t> indices, const py::array_t<double> data, 
    const py::array_t<double> x)
    {
        std::int64_t n_matvec = static_cast<std::int64_t>(x.size()); 
        py::array_t<double> y_matvec(n_matvec);
        matvec(indptr.data(), indices.data(), data.data(),x.data(), n_matvec, y_matvec.mutable_data()); 

        return y_matvec; 
    }

PYBIND11_MODULE(_kernels,m)
{
    m.doc() = R"pbdoc(
            C++ kernels for the sparse solver.

            Triangular solves for the IC(0) preconditioner and the CSR
            matrix-vector product, called from Python through pybind11.
            Arrays are read in place: index arrays must be int64 and value
            arrays float64, or pybind11 converts them, which costs a copy.
            )pbdoc"; 
    m.def("forward_solve", &forward_solve_py, py::arg("indptr") , py::arg("indices"), 
            py::arg("data"), py::arg("r"), 
            R"pbdoc(
            Solve L y = r for lower-triangular L, from the top row down.

            C++ version of CSR.forward_solve. For each row,
            y(i) = (r(i) - sum of L(i, k) y(k)) / L(i, i), with the sum over the
            stored entries before the diagonal. The diagonal must be the last
            stored entry of each row, as it is for a factor from
            incomplete_cholesky.

            Parameters
            ----------
            indptr, indices : ndarray of int64
                CSR row pointers and column indices of L.
            data : ndarray of float64
                CSR values of L.
            r : ndarray of float64, shape (n,)

            Returns
            -------
            ndarray of float64, shape (n,)
            )pbdoc"); 
    m.def("backward_solve", &backward_solve_py, py::arg("indptr") , py::arg("indices"), 
            py::arg("data"), py::arg("y") , 
            R"pbdoc(
            Solve L^T z = y, from the bottom row up.

            C++ version of CSR.backward_solve. Takes the CSR arrays of L^T
            itself, not L. L^T is upper-triangular, so the diagonal is the
            first stored entry of each row. For each row,
            z(i) = (y(i) - sum of L^T(i, k) z(k)) / L^T(i, i), with the sum over
            the entries after the diagonal.

            Parameters
            ----------
            indptr, indices : ndarray of int64
                CSR row pointers and column indices of L^T.
            data : ndarray of float64
                CSR values of L^T.
            y : ndarray of float64, shape (n,)
                Usually the output of forward_solve.

            Returns
            -------
            ndarray of float64, shape (n,)
            )pbdoc"); 
    m.def("matvec", &matvec_py, py::arg("indptr") , py::arg("indices"), 
            py::arg("data"), py::arg("x") , 
            R"pbdoc(
            Matrix-vector product K @ x for a square CSR matrix.

            C++ version of CSR.matvec. For each row, the sum of its stored
            values times x at their columns. Rows with no entries give zero.

            Parameters
            ----------
            indptr, indices : ndarray of int64
                CSR row pointers and column indices of K.
            data : ndarray of float64
                CSR values of K.
            x : ndarray of float64, shape (n,)

            Returns
            -------
            ndarray of float64, shape (n,)
            )pbdoc"); 
}
