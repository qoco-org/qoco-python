import qcos
import numpy as np
from scipy import sparse

# qcos_ext.say_hello()

if __name__ == '__main__':
    # Define problem data
    P = sparse.csc_matrix([[4, 1], [1, 2]])
    c = np.array([1, 1])
    G = sparse.csc_matrix([[1, 1], [1, 0]])
    h = np.array([1, 0.7])
    l = 2
    n = 2
    m = 2
    p = 1

    # Create an QCOS object
    prob = qcos.QCOS()
