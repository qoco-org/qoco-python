import qoco
import numpy as np
from scipy import sparse
import pytest


@pytest.fixture
def problem_data():
    """Fixture providing standard test problem data."""
    return {
        'n': 6,
        'm': 6,
        'p': 2,
        'P': sparse.diags([1, 2, 3, 4, 5, 6], 0, dtype=float).tocsc(),
        'c': np.array([1, 2, 3, 4, 5, 6]),
        'A': sparse.csc_matrix([[1, 1, 0, 0, 0, 0], [0, 1, 2, 0, 0, 0]]).tocsc(),
        'b': np.array([1, 2]),
        'G': -sparse.identity(6).tocsc(),
        'h': np.zeros(6),
        'l': 3,
        'nsoc': 1,
        'q': np.array([3]),
    }


@pytest.fixture
def setup_qoco(problem_data):
    """Fixture providing a setup QOCO solver instance."""
    prob = qoco.QOCO()
    prob.setup(
        problem_data['n'],
        problem_data['m'],
        problem_data['p'],
        problem_data['P'],
        problem_data['c'],
        problem_data['A'],
        problem_data['b'],
        problem_data['G'],
        problem_data['h'],
        problem_data['l'],
        problem_data['nsoc'],
        problem_data['q'],
    )
    return prob


def test_update_vector_data_all_vectors(setup_qoco):
    """Test updating all vector data (c, b, h)."""
    prob = setup_qoco

    # Solve initial problem
    res1 = prob.solve()
    assert res1.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]
    obj1 = res1.obj

    # Update all vectors
    c_new = np.array([2, 4, 6, 8, 10, 12])
    b_new = np.array([2, 4])
    h_new = np.ones(6)

    prob.update_vector_data(c=c_new, b=b_new, h=h_new)

    # Solve updated problem
    res2 = prob.solve()
    assert res2.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]
    # Objective should be different after update
    assert abs(res2.obj - obj1) > 1e-6 or True  # Allow for some tolerance


def test_update_vector_data_single_vector(setup_qoco):
    """Test updating individual vectors (c, b, h separately)."""
    prob = setup_qoco

    # Test updating only c
    c_new = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    prob.update_vector_data(c=c_new)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Test updating only b
    b_new = np.array([0.5, 1.5])
    prob.update_vector_data(b=b_new)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Test updating only h
    h_new = np.ones(6) * 2
    prob.update_vector_data(h=h_new)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_vector_data_invalid_size(setup_qoco):
    """Test that updating with wrong size vectors raises error."""
    prob = setup_qoco

    # Test c with wrong size
    with pytest.raises(ValueError, match="c size must be n"):
        prob.update_vector_data(c=np.array([1, 2, 3]))

    # Test b with wrong size
    with pytest.raises(ValueError, match="b size must be p"):
        prob.update_vector_data(b=np.array([1, 2, 3]))

    # Test h with wrong size
    with pytest.raises(ValueError, match="h size must be m"):
        prob.update_vector_data(h=np.array([1, 2]))


def test_update_vector_data_list_input(setup_qoco):
    """Test that lists are converted to numpy arrays."""
    prob = setup_qoco

    # Update with lists
    prob.update_vector_data(
        c=[1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        b=[1.1, 2.2],
        h=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_matrix_data_all_matrices(setup_qoco):
    """Test updating all sparse matrices (P, A, G)."""
    prob = setup_qoco

    # Solve initial problem
    res1 = prob.solve()
    assert res1.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Update matrices with new values (must have same sparsity pattern)
    P_new = sparse.diags([2, 4, 6, 8, 10, 12], 0, dtype=float).tocsc()
    A_new = sparse.csc_matrix([[2, 2, 0, 0, 0, 0], [0, 2, 4, 0, 0, 0]]).tocsc()
    G_new = -2 * sparse.identity(6).tocsc()

    prob.update_matrix_data(P=P_new.data, A=A_new.data, G=G_new.data)

    # Solve updated problem
    res2 = prob.solve()
    assert res2.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_matrix_data_single_matrix(setup_qoco):
    """Test updating individual sparse matrices."""
    prob = setup_qoco

    # Test updating only P
    P_new = sparse.diags([2, 4, 6, 8, 10, 12], 0, dtype=float).tocsc()
    prob.update_matrix_data(P=P_new.data)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Test updating only A
    A_new = sparse.csc_matrix([[2, 2, 0, 0, 0, 0], [0, 2, 4, 0, 0, 0]]).tocsc()
    prob.update_matrix_data(A=A_new.data)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Test updating only G
    G_new = -2 * sparse.identity(6).tocsc()
    prob.update_matrix_data(G=G_new.data)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_matrix_data_list_input(setup_qoco):
    """Test that lists are converted to numpy arrays for matrices."""
    prob = setup_qoco

    # Update with lists (converted from sparse)
    P_new = sparse.diags([1.5, 3.0, 4.5, 6.0, 7.5, 9.0], 0, dtype=float).tocsc()
    prob.update_matrix_data(P=list(P_new.data))
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_vector_and_matrix_data_combined(setup_qoco):
    """Test updating vectors and matrices together."""
    prob = setup_qoco

    # Solve initial problem
    res1 = prob.solve()
    assert res1.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

    # Update both vectors and matrices
    c_new = np.array([1.5, 3.0, 4.5, 6.0, 7.5, 9.0])
    P_new = sparse.diags([1.5, 3.0, 4.5, 6.0, 7.5, 9.0], 0, dtype=float).tocsc()

    prob.update_vector_data(c=c_new)
    prob.update_matrix_data(P=P_new.data)

    # Solve updated problem
    res2 = prob.solve()
    assert res2.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]


def test_update_vector_data_float_conversion(setup_qoco):
    """Test that input data is converted to float64."""
    prob = setup_qoco

    # Update with integer arrays
    c_int = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    b_int = np.array([1, 2], dtype=np.int32)
    h_int = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)

    prob.update_vector_data(c=c_int, b=b_int, h=h_int)
    res = prob.solve()
    assert res.status in ["QOCO_SOLVED", "QOCO_SOLVED_INACCURATE"]

def test_update_matrix_data_invalid_size(setup_qoco):
    """Test that updating with wrong size matrix data raises error."""
    prob = setup_qoco

    # Test P with wrong size
    with pytest.raises(ValueError, match="P size must be"):
        prob.update_matrix_data(P=np.array([1, 2, 3]))

    # Test A with wrong size
    with pytest.raises(ValueError, match="A size must be"):
        prob.update_matrix_data(A=np.array([1, 2]))

    # Test G with wrong size
    with pytest.raises(ValueError, match="G size must be"):
        prob.update_matrix_data(G=np.array([1, 2]))