import os
import shutil
import qdldl
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from qcospy.codegen_utils import *

def _generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name="qcos_custom"):
    solver_dir = output_dir + "/" + name

    print("\n")
    if os.path.exists(solver_dir):
        print("Removing existing solver.")
        shutil.rmtree(solver_dir)
    
    print("Generating solver.")
    os.mkdir(solver_dir)
    W = sparse.identity(l)
    W = sparse.csc_matrix(W)
    for qi in q:
        Wsoc = np.ones((qi, qi), dtype=np.float64)
        Wsoc = sparse.csc_matrix(Wsoc)
        W = sparse.bmat([[W, None],[None, Wsoc]])
    W = sparse.csc_matrix(W)

    Wnnz = int((W.nnz - m) / 2 + m)
    Wnnz_cnt = 0

    # Maps sparse 1D index (1,...,m^2) of W to its sparse index (1,...,Wnnz). Note that accessing an upper triangular element of W returns -1.
    Wsparse2dense = -np.ones((m*m,1))
    for j in range(m):
        for i in range(m):
            if (W[i,j] != 0.0 and i <= j):
                Wsparse2dense[i + j * m] = Wnnz_cnt
                Wnnz_cnt+=1

    # Get sparsity pattern of the regularized KKT matrix.
    reg = 1
    K = sparse.bmat([[P + reg * sparse.identity(n), A.T, G.T],[A, -reg * sparse.identity(p), None], [G, None, -W]])
    solver = qdldl.Solver(K)
    L, D, perm = solver.factors()

    N = n + m + p
    Lidx = [False for _ in range(N**2)]
    for i in range(N):
        for j in range(N):
            if (L[i,j] != 0.0):
                Lidx[j*N + i] = True

    generate_cmakelists(solver_dir)
    generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, L.nnz, Wnnz)
    Lsparse2dense = generate_ldl(n, m, p, P, A, G, perm, Lidx, Wsparse2dense, solver_dir)
    generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense)
    generate_solver(solver_dir)
    generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q)

def generate_cmakelists(solver_dir):
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("set(CMAKE_C_FLAGS \"-O3 -march=native -Wall -Wextra\")\n")
    f.write("project(qcos_custom)\n\n")
    f.write("# Build qcos_custom shared library.\n")
    f.write("add_library(qcos_custom SHARED)\n")
    f.write("target_sources(qcos_custom PRIVATE qcos_custom.c utils.c ldl.c)\n\n")
    f.write("# Build qcos demo.\n")
    f.write("add_executable(runtest runtest.c)\n")
    f.write("target_link_libraries(runtest qcos_custom)\n")
    f.close()

def generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q, Lnnz, Wnnz):
    f = open(solver_dir + "/workspace.h", "a")
    f.write("#ifndef WORKSPACE_H\n")
    f.write("#define WORKSPACE_H\n\n")

    f.write("typedef struct {\n")
    f.write("   int n;\n")
    f.write("   int m;\n")
    f.write("   int p;\n")
    f.write("   double P[%i];\n" % (len(P.data)))
    f.write("   double c[%i];\n" % (n))
    f.write("   double A[%i];\n" % (len(A.data)))
    f.write("   double b[%i];\n" % (p))
    f.write("   double G[%i];\n" % (len(G.data)))
    f.write("   double h[%i];\n" % (m))
    f.write("   int l;\n")
    f.write("   int nsoc;\n")
    f.write("   int q[%i];\n" % (len(q)))
    f.write("   double L[%i];\n" % (Lnnz))
    f.write("   double D[%i];\n" % (n + m + p))
    f.write("   double W[%i];\n" % Wnnz)
    f.write("   double kkt_reg;\n")
    f.write("} Workspace;\n\n")
    f.write("#endif")
    f.close()

def generate_ldl(n, m, p, P, A, G, perm, Lidx, Wsparse2dense, solver_dir):
    f = open(solver_dir + "/ldl.h", "a")
    f.write("#ifndef LDL_H\n")
    f.write("#define LDL_H\n\n")
    f.write("#include \"workspace.h\"\n\n")

    f.write("void ldl(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/ldl.c", "a")
    f.write("#include \"workspace.h\"\n\n")
    f.write("void ldl(Workspace* work){\n")
    N = n + m + p

    # Maps sparse 1D index (1,...,N^2) of L to its sparse index (1,...,Lnnz).
    Lsparse2dense = -np.ones(N**2)

    # Number of nonzeros of L added (Used to get sparse index of the current element under consideration).
    Lnnz = 0

    # The factorization will only access strictly lower triangular elements of L.
    for j in range(N):
        # D update.
        f.write("   work->D[%d] = " % j)
        write_Kelem(f, j, j, n, m, p, P, A, G, perm, Wsparse2dense)
        for k in range(j):
            if (Lidx[k * N + j]):
                f.write(" - work->D[%i] * " % k)
                f.write("work->L[%i] * work->L[%i]" % (Lsparse2dense[k * N + j], Lsparse2dense[k * N + j]))
        f.write(";\n")

        # L update.
        for i in range(j + 1, N):
            if (Lidx[j * N + i]):
                Lsparse2dense[j * N + i] = Lnnz
                f.write("   work->L[%i] = " % (Lnnz))
                write_Kelem(f, j, i, n, m, p, P, A, G, perm, Wsparse2dense)
                for k in range(j):
                    if (Lidx[k * N + i] and Lidx[k * N + j]):
                        f.write(" - work->L[%i] * work->L[%i] * work->D[%i]" % (Lsparse2dense[k * N + i], Lsparse2dense[k * N + j], k))
                f.write(";\n")
                f.write("   work->L[%i] /= work->D[%i];\n" % (Lnnz, j))
                Lnnz += 1
    f.write("}")
    f.close()
    return Lsparse2dense

def generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    f.write("#ifndef UTILS_H\n")
    f.write("#define UTILS_H\n\n")

    f.write("#include \"workspace.h\"\n\n")

    f.write("void load_data(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/utils.c", "a")
    f.write("#include \"utils.h\"\n\n")

    f.write("void load_data(Workspace* work){\n")
    f.write("   work->n = %d;\n" % n)
    f.write("   work->m = %d;\n" % m)
    f.write("   work->p = %d;\n" % p)

    for i in range(len(P.data)):
        f.write("   work->P[%i] = %.17g;\n" % (i, P.data[i]))
    f.write("\n")

    for i in range(len(c)):
        f.write("   work->c[%i] = %.17g;\n" % (i, c[i]))
    f.write("\n")

    for i in range(len(A.data)):
        f.write("   work->A[%i] = %.17g;\n" % (i, A.data[i]))
    f.write("\n")

    for i in range(len(b)):
        f.write("   work->b[%i] = %.17g;\n" % (i, b[i]))
    f.write("\n")

    for i in range(len(G.data)):
        f.write("   work->G[%i] = %.17g;\n" % (i, G.data[i]))
    f.write("\n")

    for i in range(len(h)):
        f.write("   work->h[%i] = %.17g;\n" % (i, h[i]))
    f.write("\n")
    f.write("   work->l = %d;\n" % l)
    f.write("   work->nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        f.write("   work->q[%i] = %d;\n" % (i, q[i]))
    f.write("\n")

    # Temporary code to set all nonzero elements in the NT scaling matrix to -1 to test ldl.
    for i in range(m*m):
        if (Wsparse2dense[i] >= 0):
            f.write("   work->W[%i] = -1.0;\n" % Wsparse2dense[i])

    f.write("   work->kkt_reg = 1;\n")
    f.write("\n")
    f.write("}")
    f.close()

def generate_solver(solver_dir):
    f = open(solver_dir + "/qcos_custom.h", "a")
    f.write("#ifndef QCOS_CUSTOM_H\n")
    f.write("#define QCOS_CUSTOM_H\n\n")
    f.write("#include \"ldl.h\"\n")
    f.write("#include \"utils.h\"\n")
    f.write("#include \"workspace.h\"\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/qcos_custom.c", "a")
    f.write("#include \"qcos_custom.h\"\n\n")
    f.close()

def generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/runtest.c", "a")
    f.write("#include <stdio.h>\n")
    f.write("#include \"qcos_custom.h\"\n\n")
    f.write("int main(){\n")
    f.write("   Workspace work;\n")
    f.write("   load_data(&work);\n")
    f.write("   ldl(&work);\n")
    f.write("   printf(\"D: {\");")
    f.write("   for(int i = 0; i < work.n + work.m + work.p; ++i){\n")
    f.write("   printf(\"%f, \", work.D[i]);\n")
    f.write("   }\n")
    f.write("   printf(\"}\\n\");\n")

    f.write("   printf(\"L: {\");")
    f.write("   for(int i = 0; i < (work.n + work.m + work.p) * (work.n + work.m + work.p); ++i){\n")
    f.write("   printf(\"%f\\n\", work.L[i]);\n")
    f.write("   }\n")
    f.write("   printf(\"}\\n\");\n")
    f.write("}")
    f.close()
