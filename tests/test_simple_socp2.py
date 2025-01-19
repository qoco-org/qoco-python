import qoco
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_simple_socp():
    # Define problem data
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6])
    G = -sparse.identity(6)
    G = G.tocsc()
    h = np.zeros(6)
    A = sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]])
    A = A.tocsc()
    b = np.array([1, 2])

    l = 0
    n = 6
    m = 6
    p = 2
    nsoc = 2
    q = np.array([3, 3])

    # Create an QOCO object.
    prob = qoco.QOCO()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    opt_obj = 5.242
    assert res.status == "QOCO_SOLVED"
    assert abs(res.obj - opt_obj) <= 1e-4
