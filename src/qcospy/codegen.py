import os
import shutil
import qdldl
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from qcospy.codegen_utils import *

def _generate_solver(n, m, p, P, c, A, b, G, h, l, nsoc, q, output_dir, name="qcosgen"):
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

    generate_cmakelists(solver_dir)
    generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q)
    generate_ldl(n, m, p, P, A, G, W, solver_dir)
    generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q)
    generate_solver(solver_dir)
    generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q)

def generate_cmakelists(solver_dir):
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("set(CMAKE_C_FLAGS \"-O3 -march=native -Wall -Wextra\")\n")
    f.write("project(qcosgen)\n\n")
    f.write("# Build qcosgen shared library.\n")
    f.write("add_library(qcosgen SHARED)\n")
    f.write("target_sources(qcosgen PRIVATE qcosgen.c utils.c ldl.c)\n\n")
    f.write("# Build qcos demo.\n")
    f.write("add_executable(runtest runtest.c)\n")
    f.write("target_link_libraries(runtest qcosgen)\n")
    f.close()

def generate_workspace(solver_dir, n, m, p, P, c, A, b, G, h, q):
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
    f.write("   double L[%i];\n" % ((n + m + p) ** 2)) # This assumes L is dense. Fix this.
    f.write("   double D[%i];\n" % (n + m + p))
    f.write("   double W[%i];\n" % m**2) # This assumes W is dense. Fix this.
    f.write("} Workspace;\n\n")
    f.write("extern Workspace work;\n")
    f.write("#endif")
    f.close()

def generate_ldl(n, m, p, P, A, G, W, solver_dir):
    f = open(solver_dir + "/ldl.h", "a")
    f.write("#ifndef LDL_H\n")
    f.write("#define LDL_H\n\n")
    f.write("void ldl();\n")
    f.write("#endif")
    f.close()

    # Get sparsity pattern of the regularized KKT matrix.
    reg = 1
    K = sparse.bmat([[P + reg * sparse.identity(n), A.T, G.T],[A, -reg * sparse.identity(p), None], [G, None, -W]])
    solver = qdldl.Solver(K)
    L, D, perm = solver.factors()

    f = open(solver_dir + "/ldl.c", "a")
    f.write("#include \"workspace.h\"\n\n")
    f.write("void ldl(){\n")
    N = n + m + p
    for j in range(N):
        f.write("   work.L[%d] = 1.0;\n" % (j * N + j))
    for j in range(N):
        # D update.
        f.write("   work.D[%d] = " % j)
        write_Kelem(f, j, j, n, m, p, P, A, G, reg)
        for k in range(j):
            f.write(" - work.D[%i] * " % k)
            f.write("work.L[%i] * work.L[%i]" % (k * N + j, k * N + j))
        f.write(";\n")

        # L update.
        for i in range(j + 1, N):
            f.write("   work.L[%i] = " % (j * N + i))
            write_Kelem(f, j, i, n, m, p, P, A, G, reg)
            for k in range(j):
                f.write(" - work.L[%i] * work.L[%i] * work.D[%i]" % (k * N + i, k * N + j, k))
            f.write(";\n")
            f.write("   work.L[%i] /= work.D[%i];\n" % (j * N + i, j))
    f.write("}")
    f.close()

def generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q):
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    f.write("#ifndef UTILS_H\n")
    f.write("#define UTILS_H\n\n")

    f.write("#include \"workspace.h\"\n\n")

    f.write("void load_data();\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/utils.c", "a")
    f.write("#include \"utils.h\"\n\n")

    f.write("void load_data(){\n")
    f.write("   work.n = %d;\n" % n)
    f.write("   work.m = %d;\n" % m)
    f.write("   work.p = %d;\n" % p)

    for i in range(len(P.data)):
        f.write("   work.P[%i] = %.17g;\n" % (i, P.data[i]))
    f.write("\n")

    for i in range(len(c)):
        f.write("   work.c[%i] = %.17g;\n" % (i, c[i]))
    f.write("\n")

    for i in range(len(A.data)):
        f.write("   work.A[%i] = %.17g;\n" % (i, A.data[i]))
    f.write("\n")

    for i in range(len(b)):
        f.write("   work.b[%i] = %.17g;\n" % (i, b[i]))
    f.write("\n")

    for i in range(len(G.data)):
        f.write("   work.G[%i] = %.17g;\n" % (i, G.data[i]))
    f.write("\n")

    for i in range(len(h)):
        f.write("   work.h[%i] = %.17g;\n" % (i, h[i]))
    f.write("\n")
    f.write("   work.l = %d;\n" % l)
    f.write("   work.nsoc = %d;\n" % nsoc)

    for i in range(len(q)):
        f.write("   work.q[%i] = %d;\n" % (i, q[i]))
    f.write("\n")

    for j in range(m):
        for i in range(m):
            if (i < 3 and j < 3 and i == j):
                f.write("   work.W[%i] = -1.0;\n" % (j*m + i))
            elif (i >= 3 and j >= 3):
                f.write("   work.W[%i] = -1.0;\n" % (j*m + i))
            else:
                f.write("   work.W[%i] = 0.0;\n" % (j*m + i))



    # for i in range(m**2):
    #     f.write("   work.W[%i] = -1.0;\n" % (i))
    f.write("\n")
    f.write("}")
    f.close()

def generate_solver(solver_dir):
    f = open(solver_dir + "/qcosgen.h", "a")
    f.write("#ifndef QCOSGEN_H\n")
    f.write("#define QCOSGEN_H\n\n")
    f.write("#include \"ldl.h\"\n")
    f.write("#include \"utils.h\"\n")
    f.write("#include \"workspace.h\"\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/qcosgen.c", "a")
    f.write("#include \"qcosgen.h\"\n\n")
    f.close()

def generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/runtest.c", "a")
    f.write("#include <stdio.h>\n")
    f.write("#include \"qcosgen.h\"\n\n")
    f.write("Workspace work;\n")
    f.write("int main(){\n")
    f.write("   load_data();\n")
    f.write("   ldl();\n")
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
