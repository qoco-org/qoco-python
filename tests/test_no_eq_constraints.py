import qoco
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_linear_objective():

    n = 6
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6])

    p = 0
    A = None
    b = None

    l = 3
    m = 6
    nsoc = 1
    q = np.array([3])
    G = -sparse.identity(m)
    G = G.tocsc()
    h = np.zeros(m)

    # Create an QOCO object.
    prob = qoco.QOCO()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    opt_obj = -0.758
    assert res.status == "QOCO_SOLVED"
    assert abs(res.obj - opt_obj) <= 1e-4
