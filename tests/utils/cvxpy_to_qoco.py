import cvxpy as cp
from scipy import sparse
from numpy import int32


def convert(prob):
    data, _, _ = prob.get_problem_data(cp.CLARABEL)
    p = data["dims"].zero
    l = data["dims"].nonneg
    q = data["dims"].soc
    m = l + sum(q)
    nsoc = len(q)

    c = data["c"]
    try:
        P = data["P"]
        P = sparse.triu(P, format="csc")
    except:
        P = None

    n = len(c)
    A = data["A"][0:p, :]
    b = data["b"][0:p]

    G = data["A"][p::, :]
    h = data["b"][p::]

    if P is not None:
        P.indices = P.indices.astype(int32)
        P.indptr = P.indptr.astype(int32)
    if A is not None:
        A.indices = A.indices.astype(int32)
        A.indptr = A.indptr.astype(int32)
    if G is not None:
        G.indices = G.indices.astype(int32)
        G.indptr = G.indptr.astype(int32)

    return n, m, p, P, c, A, b, G, h, l, nsoc, q
