import qcospy as qcos
from scipy import sparse
import numpy as np

P = sparse.diags([1,2,3,4,5,6], 0)
P = P.tocsc()

c = np.array([1,2,3,4,5,6])
G = -sparse.identity(6)
G = G.tocsc()
h = np.zeros(6)
A = sparse.csc_matrix([[1,1,0,0,0,0],[0,1,2,0,0,0]])
A = A.tocsc()
b = np.array([1, 2])

l = 3
n = 6
m = 6
p = 2
nsoc = 1
q = np.array([3])

# Create an QCOS object.
prob = qcos.QCOS()

# Setup workspace.
prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

prob.generate_solver()

# prob.update_settings(verbose=1)

# # Solve problem.
# res = prob.solve()

# opt_obj = 4.042
# assert(res.status == 'QCOS_SOLVED')
# assert(abs(res.obj - opt_obj) <= 1e-4)
