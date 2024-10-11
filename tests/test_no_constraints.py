import qoco
import numpy as np
from scipy import sparse
from tests.utils.run_generated_solver import *


def test_no_constraints():

    n = 6
    P = sparse.diags([1, 2, 3, 4, 5, 6], 0)
    P = P.tocsc()

    c = np.array([1, 2, 3, 4, 5, 6])

    p = 0
    A = None
    b = None

    l = 0
    m = 0
    nsoc = 0
    q = None
    G = None
    h = None

    # Create an QOCO object.
    prob = qoco.QOCO()

    # Setup workspace.
    prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

    # Solve problem.
    res = prob.solve()

    prob.generate_solver("tests/", "qoco_custom_no_cons")
    codegen_solved, codegen_obj, average_runtime_ms = run_generated_solver(
        "tests/qoco_custom_no_cons"
    )

    opt_obj = -10.5
    assert res.status == "QOCO_SOLVED"
    assert abs(res.obj - opt_obj) <= 1e-4
    assert codegen_solved == 1
    assert abs(codegen_obj - opt_obj) <= 1e-4
