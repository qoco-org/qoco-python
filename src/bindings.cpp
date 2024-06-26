#include <pybind11/pybind11.h>

extern "C"
{
    void say_hello();
}

namespace py = pybind11;

PYBIND11_MODULE(qcos_ext, m)
{
    m.def("say_hello", &say_hello, "A function that prints 'Hello, World!'");
}
