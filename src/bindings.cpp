#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "qcos.h"

namespace py = pybind11;
using namespace pybind11::literals;

class CSC
{
public:
    CSC(py::object A);
    ~CSC();
    QCOSCscMatrix &getcsc() const;
    py::array_t<QCOSFloat> _x;
    py::array_t<QCOSInt> _i;
    py::array_t<QCOSInt> _p;
    QCOSInt m;
    QCOSInt n;
    QCOSInt nnz;

private:
    QCOSCscMatrix *_csc;
};

CSC::CSC(py::object A)
{
    py::object spa = py::module::import("scipy.sparse");

    py::tuple dim = A.attr("shape");
    int m = dim[0].cast<int>();
    int n = dim[1].cast<int>();

    if (!spa.attr("isspmatrix_csc")(A))
    {
        A = spa.attr("csc_matrix")(A);
    }

    this->_csc = new QCOSCscMatrix();
    this->_csc->m = m;
    this->_csc->n = n;
    this->_csc->x = (QCOSFloat *)this->_x.data();
    this->_csc->i = (QCOSInt *)this->_i.data();
    this->_csc->p = (QCOSInt *)this->_p.data();
    this->_csc->nnz = A.attr("nnz").cast<int>();

    this->m = this->_csc->m;
    this->n = this->_csc->n;
    this->nnz = this->_csc->nnz;
}

QCOSCscMatrix &CSC::getcsc() const
{
    return *this->_csc;
}

CSC::~CSC()
{
    delete this->_csc;
}

class PyQCOSSolution
{
public:
    PyQCOSSolution(QCOSSolution &, QCOSInt, QCOSInt, QCOSInt);
    py::array_t<QCOSFloat> get_x();
    py::array_t<QCOSFloat> get_s();
    py::array_t<QCOSFloat> get_y();
    py::array_t<QCOSFloat> get_z();

private:
    QCOSInt _n;
    QCOSInt _m;
    QCOSInt _p;
    QCOSSolution &_solution;
};

PyQCOSSolution::PyQCOSSolution(QCOSSolution &solution, QCOSInt n, QCOSInt m, QCOSInt p) : _n(n), _m(m), _p(p), _solution(solution)
{
}

py::array_t<QCOSFloat> PyQCOSSolution::get_x()
{
    return py::array_t<QCOSFloat>(
        {this->_n},
        {sizeof(QCOSFloat)},
        this->_solution.x);
}

py::array_t<QCOSFloat> PyQCOSSolution::get_s()
{
    return py::array_t<QCOSFloat>(
        {this->_m},
        {sizeof(QCOSFloat)},
        this->_solution.s);
}

py::array_t<QCOSFloat> PyQCOSSolution::get_y()
{
    return py::array_t<QCOSFloat>(
        {this->_p},
        {sizeof(QCOSFloat)},
        this->_solution.y);
}

py::array_t<QCOSFloat> PyQCOSSolution::get_z()
{
    return py::array_t<QCOSFloat>(
        {this->_m},
        {sizeof(QCOSFloat)},
        this->_solution.z);
}

class PyQCOSSolver
{
}

extern "C"
{
    void say_hello();
}

PYBIND11_MODULE(qcos_ext, m)
{
    m.def("say_hello", &say_hello, "A function that prints 'Hello, World!'");
}
