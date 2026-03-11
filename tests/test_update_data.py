import qoco
import numpy as np
from scipy.sparse import csc_matrix
import pytest

abstol = 1e-10
reltol = 1e-10


@pytest.fixture
def problem_data():
    """Fixture providing standard test problem data."""
    # problem dimensions
    n = 6
    m = 6
    p = 2
    l = 3
    nsoc = 1
    
    # initial problem data
    P = csc_matrix((
            [1, 2, 3, 4, 5, 6], 
            [0, 1, 2, 3, 4, 5], 
            [0, 1, 2, 3, 4, 5, 6]
        ), 
        shape=(n, n),
        dtype=float,
    )

    A = csc_matrix((
            [1, 1, 1, 2], 
            [0, 0, 1, 1], 
            [0, 1, 3, 4, 4, 4, 4]
        ), 
        shape=(p, n),
        dtype=float,
    )

    G = csc_matrix((
            [-1, -1, -1, -1, -1, -1], 
            [0, 1, 2, 3, 4, 5], 
            [0, 1, 2, 3, 4, 5, 6]
        ), 
        shape=(m, n),
        dtype=float,
    )

    c = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    b = np.array([1, 2], dtype=float)
    h = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    q = np.array([3], dtype=float)

    return {
        "n": n,
        "m": m,
        "p": p,
        "P": P,
        "c": c,
        "A": A,
        "b": b,
        "G": G,
        "h": h,
        "l": l,
        "nsoc": nsoc,
        "q": q,
    }


@pytest.fixture
def problem(problem_data):
    """Fixture providing a setup QOCO solver instance."""
    prob = qoco.QOCO()
    prob.setup(
        problem_data["n"],
        problem_data["m"],
        problem_data["p"],
        problem_data["P"],
        problem_data["c"],
        problem_data["A"],
        problem_data["b"],
        problem_data["G"],
        problem_data["h"],
        problem_data["l"],
        problem_data["nsoc"],
        problem_data["q"],
        abstol=abstol, 
        reltol=reltol,
    )
    return prob


def test_update_vector_data(problem_data, problem):    
    # setup and solve initial problem
    res1 = problem.solve()
    assert res1.status == "QOCO_SOLVED"

    # new vector data
    c_new = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    b_new = np.array([4, 5], dtype=float)
    h_new = np.array([1, 1, 1, 1, 1, 1], dtype=float)

    # update solver and solve again
    problem.update_vector_data(c=c_new, b=b_new, h=h_new)
    res2 = problem.solve()
    assert res2.status == "QOCO_SOLVED"

    # reference solution
    prob2 = qoco.QOCO()
    prob2.setup(
        problem_data["n"],
        problem_data["m"],
        problem_data["p"],
        problem_data["P"],
        c_new,
        problem_data["A"],
        b_new,
        problem_data["G"],
        h_new,
        problem_data["l"],
        problem_data["nsoc"],
        problem_data["q"],
        abstol=abstol, 
        reltol=reltol,
    )
    res2_ref = prob2.solve()
    assert res2_ref.status == "QOCO_SOLVED"

    assert np.allclose(res2.x, res2_ref.x)
    assert np.allclose(res2.s, res2_ref.s)
    assert np.allclose(res2.y, res2_ref.y)
    assert np.allclose(res2.z, res2_ref.z)
    assert np.allclose(res2.obj, res2_ref.obj)
    assert np.allclose(res2.dres, res2_ref.dres)
    assert np.allclose(res2.pres, res2_ref.pres)
    assert np.allclose(res2.gap, res2_ref.gap)
    assert np.allclose(res2.iters, res2_ref.iters)


def test_update_cost_matrix(problem_data, problem):    
    # setup and solve initial problem
    res1 = problem.solve()
    assert res1.status == "QOCO_SOLVED"

    # new matrix data
    P_data_new = np.array([6, 5, 4, 3, 2, 1], dtype=float)
    P_new = problem_data["P"].copy()
    P_new.data = P_data_new.copy()

    # update solver and solve again
    problem.update_matrix_data(P=P_data_new)
    res2 = problem.solve()
    assert res2.status == "QOCO_SOLVED"

    # reference solution
    prob2 = qoco.QOCO()
    prob2.setup(
        problem_data["n"],
        problem_data["m"],
        problem_data["p"],
        P_new,
        problem_data["c"],
        problem_data["A"],
        problem_data["b"],
        problem_data["G"],
        problem_data["h"],
        problem_data["l"],
        problem_data["nsoc"],
        problem_data["q"],
        abstol=abstol, 
        reltol=reltol,
    )
    res2_ref = prob2.solve()
    assert res2_ref.status == "QOCO_SOLVED"

    assert np.allclose(res2.x, res2_ref.x)
    assert np.allclose(res2.s, res2_ref.s)
    assert np.allclose(res2.y, res2_ref.y)
    assert np.allclose(res2.z, res2_ref.z)
    assert np.allclose(res2.obj, res2_ref.obj)
    assert np.allclose(res2.dres, res2_ref.dres)
    assert np.allclose(res2.pres, res2_ref.pres)
    assert np.allclose(res2.gap, res2_ref.gap)
    assert np.allclose(res2.iters, res2_ref.iters)


def test_update_constraint_matrix(problem_data, problem):    
    # setup and solve initial problem
    res1 = problem.solve()
    assert res1.status == "QOCO_SOLVED"

    # new matrix data
    A_data_new = np.array([1, 2, 3, 4], dtype=float)
    A_new = problem_data["A"].copy()
    A_new.data = A_data_new.copy()

    # update solver and solve again
    problem.update_matrix_data(A=A_data_new)
    res2 = problem.solve()
    assert res2.status == "QOCO_SOLVED"

    # reference solution
    prob2 = qoco.QOCO()
    prob2.setup(
        problem_data["n"],
        problem_data["m"],
        problem_data["p"],
        problem_data["P"],
        problem_data["c"],
        A_new,
        problem_data["b"],
        problem_data["G"],
        problem_data["h"],
        problem_data["l"],
        problem_data["nsoc"],
        problem_data["q"],
        abstol=abstol, 
        reltol=reltol,
    )
    res2_ref = prob2.solve()
    assert res2_ref.status == "QOCO_SOLVED"

    assert np.allclose(res2.x, res2_ref.x)
    assert np.allclose(res2.s, res2_ref.s)
    assert np.allclose(res2.y, res2_ref.y)
    assert np.allclose(res2.z, res2_ref.z)
    assert np.allclose(res2.obj, res2_ref.obj)
    assert np.allclose(res2.dres, res2_ref.dres)
    assert np.allclose(res2.pres, res2_ref.pres)
    assert np.allclose(res2.gap, res2_ref.gap)
    assert np.allclose(res2.iters, res2_ref.iters)


def test_update_soc_matrix(problem_data, problem):    
    # setup and solve initial problem
    res1 = problem.solve()
    assert res1.status == "QOCO_SOLVED"

    # new matrix data
    G_data_new = np.array([-2, -2, -2, -2, -2, -2], dtype=float)
    G_new = problem_data["G"].copy()
    G_new.data = G_data_new.copy()

    # update solver and solve again
    problem.update_matrix_data(G=G_data_new)
    res2 = problem.solve()
    assert res2.status == "QOCO_SOLVED"

    # reference solution
    prob2 = qoco.QOCO()
    prob2.setup(
        problem_data["n"],
        problem_data["m"],
        problem_data["p"],
        problem_data["P"],
        problem_data["c"],
        problem_data["A"],
        problem_data["b"],
        G_new,
        problem_data["h"],
        problem_data["l"],
        problem_data["nsoc"],
        problem_data["q"],
        abstol=abstol, 
        reltol=reltol,
    )
    res2_ref = prob2.solve()
    assert res2_ref.status == "QOCO_SOLVED"

    assert np.allclose(res2.x, res2_ref.x)
    assert np.allclose(res2.s, res2_ref.s)
    assert np.allclose(res2.y, res2_ref.y)
    assert np.allclose(res2.z, res2_ref.z)
    assert np.allclose(res2.obj, res2_ref.obj)
    assert np.allclose(res2.dres, res2_ref.dres)
    assert np.allclose(res2.pres, res2_ref.pres)
    assert np.allclose(res2.gap, res2_ref.gap)
    assert np.allclose(res2.iters, res2_ref.iters)
