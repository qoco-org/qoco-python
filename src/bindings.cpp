/* Copyright (c) 2024, Govind M. Chari <govindchari1@gmail.com> */
/* This source code is licensed under the BSD 3-Clause License  */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "qoco.h"

namespace py = pybind11;
using namespace pybind11::literals;

class CSC
{
public:
    CSC(py::object A);
    ~CSC();
    QOCOCscMatrix *getcsc() const;
    py::array_t<QOCOFloat> _x;
    py::array_t<QOCOInt> _i;
    py::array_t<QOCOInt> _p;
    QOCOInt m;
    QOCOInt n;
    QOCOInt nnz;

private:
    QOCOCscMatrix *_csc;
};

CSC::CSC(py::object A)
{
    py::object spa = py::module::import("scipy.sparse");

    if (A == py::none())
    {
        this->m = 0;
        this->n = 0;
        this->nnz = 0;
        this->_csc = nullptr;
    }

    else
    {
        py::tuple dim = A.attr("shape");

        int m = dim[0].cast<int>();
        int n = dim[1].cast<int>();

        if (!spa.attr("isspmatrix_csc")(A))
        {
            A = spa.attr("csc_matrix")(A);
        }

        this->_p = A.attr("indptr").cast<py::array_t<QOCOInt, py::array::c_style>>();
        this->_i = A.attr("indices").cast<py::array_t<QOCOInt, py::array::c_style>>();
        this->_x = A.attr("data").cast<py::array_t<QOCOFloat, py::array::c_style>>();

        this->_csc = new QOCOCscMatrix();
        this->_csc->m = m;
        this->_csc->n = n;
        this->_csc->x = (QOCOFloat *)this->_x.data();
        this->_csc->i = (QOCOInt *)this->_i.data();
        this->_csc->p = (QOCOInt *)this->_p.data();
        this->_csc->nnz = A.attr("nnz").cast<int>();

        this->m = this->_csc->m;
        this->n = this->_csc->n;
        this->nnz = this->_csc->nnz;
    }
}

QOCOCscMatrix *CSC::getcsc() const
{
    return this->_csc;
}

CSC::~CSC()
{
    if (this->_csc)
        delete this->_csc;
}

class PyQOCOSolution
{
public:
    PyQOCOSolution(QOCOSolution &, QOCOInt, QOCOInt, QOCOInt);
    py::array_t<QOCOFloat> get_x();
    py::array_t<QOCOFloat> get_s();
    py::array_t<QOCOFloat> get_y();
    py::array_t<QOCOFloat> get_z();
    QOCOSolution &_solution;

private:
    QOCOInt _n;
    QOCOInt _m;
    QOCOInt _p;
};

PyQOCOSolution::PyQOCOSolution(QOCOSolution &solution, QOCOInt n, QOCOInt m, QOCOInt p) : _n(n), _m(m), _p(p), _solution(solution)
{
}

py::array_t<QOCOFloat> PyQOCOSolution::get_x()
{
    return py::array_t<QOCOFloat>(
        {this->_n},
        {sizeof(QOCOFloat)},
        this->_solution.x);
}

py::array_t<QOCOFloat> PyQOCOSolution::get_s()
{
    return py::array_t<QOCOFloat>(
        {this->_m},
        {sizeof(QOCOFloat)},
        this->_solution.s);
}

py::array_t<QOCOFloat> PyQOCOSolution::get_y()
{
    return py::array_t<QOCOFloat>(
        {this->_p},
        {sizeof(QOCOFloat)},
        this->_solution.y);
}

py::array_t<QOCOFloat> PyQOCOSolution::get_z()
{
    return py::array_t<QOCOFloat>(
        {this->_m},
        {sizeof(QOCOFloat)},
        this->_solution.z);
}

class PyQOCOSolver
{
public:
    PyQOCOSolver(QOCOInt, QOCOInt, QOCOInt, const CSC &, const py::array_t<QOCOFloat>, const CSC &, const py::array_t<QOCOFloat>, const CSC &, const py::array_t<QOCOFloat>, QOCOInt, QOCOInt, const py::array_t<QOCOInt>, QOCOSettings *);
    ~PyQOCOSolver();

    QOCOSettings *get_settings();
    PyQOCOSolution &get_solution();

    QOCOInt update_settings(const QOCOSettings &);
    // QOCOInt update_vector_data(py::object, py::object, py::object);
    // QOCOInt update_matrix_data(py::object, py::object, py::object);

    QOCOInt solve();

private:
    QOCOInt n;
    QOCOInt m;
    QOCOInt p;

    const CSC &_P;
    py::array_t<QOCOFloat> _c;

    const CSC &_A;
    py::array_t<QOCOFloat> _b;

    const CSC &_G;
    py::array_t<QOCOFloat> _h;

    QOCOInt l;
    QOCOInt nsoc;
    py::array_t<QOCOInt> _q;

    QOCOSolver *_solver;
};

PyQOCOSolver::PyQOCOSolver(QOCOInt n, QOCOInt m, QOCOInt p, const CSC &P, const py::array_t<QOCOFloat> c, const CSC &A, const py::array_t<QOCOFloat> b, const CSC &G, const py::array_t<QOCOFloat> h, QOCOInt l, QOCOInt nsoc, const py::array_t<QOCOInt> q, QOCOSettings *settings) : n(n), m(m), p(p), _P(P), _c(c), _A(A), _b(b), _G(G), _h(h), l(l), nsoc(nsoc), _q(q)
{
    this->_solver = new QOCOSolver();

    py::tuple dim = this->_b.attr("shape");
    QOCOFloat *bptr;
    if (dim[0].cast<int>() == 0)
    {
        bptr = (QOCOFloat *)nullptr;
    }
    else
    {
        bptr = (QOCOFloat *)this->_b.data();
    }

    dim = this->_h.attr("shape");
    QOCOFloat *hptr;
    if (dim[0].cast<int>() == 0)
    {
        hptr = nullptr;
    }
    else
    {
        hptr = (QOCOFloat *)this->_h.data();
    }

    dim = this->_q.attr("shape");
    QOCOInt *qptr;
    if (dim[0].cast<int>() == 0)
    {
        qptr = nullptr;
    }
    else
    {
        qptr = (QOCOInt *)this->_q.data();
    }

    QOCOInt status = qoco_setup(this->_solver, n, m, p, this->_P.getcsc(), (QOCOFloat *)this->_c.data(), this->_A.getcsc(), bptr, this->_G.getcsc(), hptr, l, nsoc, qptr, settings);

    if (status)
    {
        std::string message = "Setup Error (Error Code " + std::to_string(status) + ")";
        throw py::value_error(message);
    }
}

PyQOCOSolver::~PyQOCOSolver()
{
    qoco_cleanup(this->_solver);
}

QOCOSettings *PyQOCOSolver::get_settings()
{
    return this->_solver->settings;
}

PyQOCOSolution &PyQOCOSolver::get_solution()
{
    PyQOCOSolution *solution = new PyQOCOSolution(*this->_solver->sol, this->n, this->m, this->p);
    return *solution;
}

QOCOInt PyQOCOSolver::solve()
{
    py::gil_scoped_release release;
    QOCOInt status = qoco_solve(this->_solver);
    py::gil_scoped_acquire acquire;
    return status;
}

QOCOInt PyQOCOSolver::update_settings(const QOCOSettings &new_settings)
{
    return qoco_update_settings(this->_solver, &new_settings);
}

PYBIND11_MODULE(qoco_ext, m)
{
    // Enums.
    py::enum_<qoco_solve_status>(m, "qoco_solve_status", py::module_local())
        .value("QOCO_UNSOLVED", QOCO_UNSOLVED)
        .value("QOCO_SOLVED", QOCO_SOLVED)
        .value("QOCO_SOLVED_INACCURATE", QOCO_SOLVED_INACCURATE)
        .value("QOCO_MAX_ITER", QOCO_MAX_ITER)
        .value("QOCO_NUMERICAL_ERROR", QOCO_NUMERICAL_ERROR)
        .export_values();

    // CSC.
    py::class_<CSC>(m, "CSC", py::module_local())
        .def(py::init<py::object>())
        .def_readonly("m", &CSC::m)
        .def_readonly("n", &CSC::n)
        .def_readonly("x", &CSC::_x)
        .def_readonly("i", &CSC::_i)
        .def_readonly("p", &CSC::_p)
        .def_readonly("nnz", &CSC::nnz);

    // Settings.
    py::class_<QOCOSettings>(m, "QOCOSettings", py::module_local())
        .def(py::init([]()
                      { return new QOCOSettings(); }))
        .def_readwrite("max_iters", &QOCOSettings::max_iters)
        .def_readwrite("bisect_iters", &QOCOSettings::bisect_iters)
        .def_readwrite("ruiz_iters", &QOCOSettings::ruiz_iters)
        .def_readwrite("iter_ref_iters", &QOCOSettings::iter_ref_iters)
        .def_readwrite("abstol", &QOCOSettings::abstol)
        .def_readwrite("reltol", &QOCOSettings::reltol)
        .def_readwrite("abstol_inacc", &QOCOSettings::abstol_inacc)
        .def_readwrite("reltol_inacc", &QOCOSettings::reltol_inacc)
        .def_readwrite("kkt_static_reg", &QOCOSettings::kkt_static_reg)
        .def_readwrite("kkt_dynamic_reg", &QOCOSettings::kkt_dynamic_reg)
        .def_readwrite("verbose", &QOCOSettings::verbose);

    m.def("set_default_settings", &set_default_settings);

    // Solution.
    py::class_<PyQOCOSolution>(m, "QOCOSolution", py::module_local())
        .def_property_readonly("x", &PyQOCOSolution::get_x)
        .def_property_readonly("s", &PyQOCOSolution::get_s)
        .def_property_readonly("y", &PyQOCOSolution::get_y)
        .def_property_readonly("z", &PyQOCOSolution::get_z)
        .def_property_readonly("iters", [](const PyQOCOSolution &sol)
                               { return sol._solution.iters; })
        .def_property_readonly("setup_time_sec", [](const PyQOCOSolution &sol)
                               { return sol._solution.setup_time_sec; })
        .def_property_readonly("solve_time_sec", [](const PyQOCOSolution &sol)
                               { return sol._solution.solve_time_sec; })
        .def_property_readonly("obj", [](const PyQOCOSolution &sol)
                               { return sol._solution.obj; })
        .def_property_readonly("pres", [](const PyQOCOSolution &sol)
                               { return sol._solution.pres; })
        .def_property_readonly("dres", [](const PyQOCOSolution &sol)
                               { return sol._solution.dres; })
        .def_property_readonly("gap", [](const PyQOCOSolution &sol)
                               { return sol._solution.gap; })
        .def_property_readonly("status", [](const PyQOCOSolution &sol)
                               { return sol._solution.status; });

    // Solver.
    py::class_<PyQOCOSolver>(m, "QOCOSolver", py::module_local())
        .def(py::init<QOCOInt, QOCOInt, QOCOInt, const CSC &, const py::array_t<QOCOFloat>, const CSC &, const py::array_t<QOCOFloat>, const CSC &, const py::array_t<QOCOFloat>, QOCOInt, QOCOInt, const py::array_t<QOCOInt>, QOCOSettings *>(), "n"_a, "m"_a, "p"_a, "P"_a, "c"_a.noconvert(), "A"_a, "b"_a.noconvert(), "G"_a, "h"_a.noconvert(), "l"_a, "nsoc"_a, "q"_a.noconvert(), "settings"_a)
        .def_property_readonly("solution", &PyQOCOSolver::get_solution, py::return_value_policy::reference)
        .def("update_settings", &PyQOCOSolver::update_settings)
        .def("solve", &PyQOCOSolver::solve)
        .def("get_settings", &PyQOCOSolver::get_settings, py::return_value_policy::reference);
}
