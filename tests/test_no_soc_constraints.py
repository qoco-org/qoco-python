import qcospy as qcos
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *

def test_linear_objective():

    n = 6
    P = sparse.diags([1,2,3,4,5,6], 0)
    P = P.tocsc()

    c = np.array([1,2,3,4,5,6])

    p = 2
    A = sparse.csc_matrix([[1,1,0,0,0,0],[0,1,2,0,0,0]])
    b = np.array([1,2])

    l = 6
    m = 6
    nsoc = 0
    q = None
    G = -sparse.identity(m)
    G = G.tocsc()
    h = np.zeros(m)


    # Create an QCOS object.
    prob = qcos.QCOS()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    prob.generate_solver("tests/", "qcos_custom_no_soc")
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver("tests/qcos_custom_no_soc")

    opt_obj = 4.800
    assert(res.status == 'QCOS_SOLVED')
    assert(abs(res.obj - opt_obj) <= 1e-4)
    assert(codegen_solved)
    assert(abs(codegen_obj - opt_obj) <= 1e-4)
