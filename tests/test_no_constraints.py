import qcospy as qcos

import numpy as np
from scipy import sparse

def test_linear_objective():

    n = 6
    P = sparse.diags([1,2,3,4,5,6], 0)
    P = P.tocsc()

    c = np.array([1,2,3,4,5,6])

    p = 0
    A = None
    b = None

    l = 0
    m = 0
    nsoc = 0
    q = None
    G = None
    h = None


    # Create an QCOS object.
    prob = qcos.QCOS()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    opt_obj = -10.5
    assert(res.status == 'QCOS_SOLVED')
    assert(abs(res.obj - opt_obj) <= 1e-4)
