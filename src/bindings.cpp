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
public:
    PyQCOSSolver(QCOSInt, QCOSInt, QCOSInt, const CSC &, const py::array_t<QCOSFloat>, const CSC &, const py::array_t<QCOSFloat>, const CSC &, const py::array_t<QCOSFloat>, QCOSInt, QCOSInt, const py::array_t<QCOSInt>, QCOSSettings *);
    ~PyQCOSSolver();

    QCOSSettings *get_settings();
    PyQCOSSolution &get_solution();

    // QCOSInt update_settings(const QCOSSettings &);
    // QCOSInt update_vector_data(py::object, py::object, py::object);
    // QCOSInt update_matrix_data(py::object, py::object, py::object);

    QCOSInt solve();

private:
    QCOSInt n;
    QCOSInt m;
    QCOSInt p;

    const CSC &_P;
    py::array_t<QCOSFloat> _c;

    const CSC &_A;
    py::array_t<QCOSFloat> _b;

    const CSC &_G;
    py::array_t<QCOSFloat> _h;

    QCOSInt l;
    QCOSInt nsoc;
    py::array_t<QCOSInt> _q;

    QCOSSolver *_solver;
};

PyQCOSSolver::PyQCOSSolver(QCOSInt n, QCOSInt m, QCOSInt p, const CSC &P, const py::array_t<QCOSFloat> c, const CSC &A, const py::array_t<QCOSFloat> b, const CSC &G, const py::array_t<QCOSFloat> h, QCOSInt l, QCOSInt nsoc, const py::array_t<QCOSInt> q, QCOSSettings *settings) : n(n), m(m), p(p), _P(P), _c(c), _A(A), _b(b), _G(G), _h(h), l(l), nsoc(nsoc), _q(q)
{
    this->_solver = new QCOSSolver();
    QCOSInt status = qcos_setup(this->_solver, n, m, p, &this->_P.getcsc(), (QCOSFloat *)this->_c.data(), &this->_A.getcsc(), (QCOSFloat *)this->_b.data(), &this->_G.getcsc(), (QCOSFloat *)this->_h.data(), l, nsoc, (QCOSInt *)this->_q.data(), settings);

    if (status)
    {
        std::string message = "Setup Error (Error Code " + std::to_string(status) + ")";
        throw py::value_error(message);
    }
}

PyQCOSSolver::~PyQCOSSolver()
{
    qcos_cleanup(this->_solver);
}

QCOSSettings *PyQCOSSolver::get_settings()
{
    return this->_solver->settings;
}

PyQCOSSolution &PyQCOSSolver::get_solution()
{
    PyQCOSSolution *solution = new PyQCOSSolution(*this->_solver->sol, this->n, this->m, this->p);
    return *solution;
}

QCOSInt PyQCOSSolver::solve()
{
    py::gil_scoped_release release;
    QCOSInt status = qcos_solve(this->_solver);
    py::gil_scoped_acquire acquire;
    return status;
}

extern "C"
{
    void say_hello();
}

PYBIND11_MODULE(qcos_ext, m)
{
    m.def("say_hello", &say_hello, "A function that prints 'Hello, World!'");

    // Enums.
    py::enum_<qcos_solve_status>(m, "qcos_solve_status", py::module_local())
        .value("QCOS_UNSOLVED", QCOS_UNSOLVED)
        .value("QCOS_SOLVED", QCOS_SOLVED)
        .value("QCOS_SOLVED_INACCURATE", QCOS_SOLVED_INACCURATE)
        .value("QCOS_MAX_ITER", QCOS_MAX_ITER)
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
    py::class_<QCOSSettings>(m, "QCOSSettings", py::module_local())
        .def(py::init([]()
                      { return new QCOSSettings(); }))
        .def_readwrite("max_iters", &QCOSSettings::max_iters)
        .def_readwrite("bisection_iters", &QCOSSettings::bisection_iters)
        .def_readwrite("ruiz_iters", &QCOSSettings::ruiz_iters)
        .def_readwrite("iterative_refinement_iterations", &QCOSSettings::iterative_refinement_iterations)
        .def_readwrite("abstol", &QCOSSettings::abstol)
        .def_readwrite("reltol", &QCOSSettings::reltol)
        .def_readwrite("abstol_inaccurate", &QCOSSettings::abstol_inaccurate)
        .def_readwrite("reltol_inaccurate", &QCOSSettings::reltol_inaccurate)
        .def_readwrite("reg", &QCOSSettings::reg)
        .def_readwrite("verbose", &QCOSSettings::verbose);

    // Solution.
    py::class_<PyQCOSSolution>(m, "QCOSSolution", py::module_local())
        .def_property_readonly("x", &PyQCOSSolution::get_x)
        .def_property_readonly("s", &PyQCOSSolution::get_s)
        .def_property_readonly("y", &PyQCOSSolution::get_y)
        .def_property_readonly("z", &PyQCOSSolution::get_z);

    // Solver.
    py::class_<PyQCOSSolver>(m, "QCOSSolver", py::module_local())
        .def(py::init<QCOSInt, QCOSInt, QCOSInt, const CSC &, const py::array_t<QCOSFloat>, const CSC &, const py::array_t<QCOSFloat>, const CSC &, const py::array_t<QCOSFloat>, QCOSInt, QCOSInt, const py::array_t<QCOSInt>, QCOSSettings *>(), "n"_a, "m"_a, "p"_a, "P"_a, "c"_a.noconvert(), "A"_a, "b"_a.noconvert(), "G"_a, "h"_a.noconvert(), "l"_a, "nsoc"_a, "q"_a.noconvert(), "settings"_a)
        .def_property_readonly("solution", &PyQCOSSolver::get_solution, py::return_value_policy::reference)
        .def("solve", &PyQCOSSolver::solve)
        .def("get_settings", &PyQCOSSolver::get_settings, py::return_value_policy::reference);
}
