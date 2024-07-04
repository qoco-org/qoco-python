import pytest
import qcos
import numpy as np
from scipy import sparse

# Define problem data
n = 2
P = None
c = np.array([-1, -2])

p = 0
A = None
b = None

l = 3
m = 3
nsoc = 0
q = None
G = sparse.csc_matrix([[-1, 0],[0, -1],[1, 1]])
h = np.array([0,0,1])

# Create an QCOS object.
prob = qcos.QCOS()

# Setup workspace.
prob.setup(n, m, p, P, c, A, b, G, h, l, nsoc, q)

prob.update_settings(verbose=1)

# Solve problem.
res = prob.solve()

opt_obj = -2.000

assert(abs(res.obj - opt_obj) <= 1e-4)
