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

    # Maps sparse 1D index (1,...,m^2) of W to its sparse index (1,...,Wnnz). Note that accessing an lower triangular element of W returns -1.
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
    Lsparse2dense = generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense)
    generate_cone(solver_dir, m, Wnnz, Wsparse2dense)
    generate_kkt(solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense)
    generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense)
    generate_solver(solver_dir, m, Wsparse2dense)
    generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q)

def generate_cmakelists(solver_dir):
    f = open(solver_dir + "/CMakeLists.txt", "a")
    f.write("cmake_minimum_required(VERSION 3.18)\n")
    f.write("set(CMAKE_C_FLAGS \"-O3 -march=native -Wall -Wextra\")\n")
    f.write("project(qcos_custom)\n\n")
    f.write("# Build qcos_custom shared library.\n")
    f.write("add_library(qcos_custom SHARED)\n")
    f.write("target_sources(qcos_custom PRIVATE qcos_custom.c cone.c utils.c ldl.c kkt.c)\n\n")
    f.write("target_link_libraries(qcos_custom m)")
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
    f.write("   double x[%i];\n" % (n))
    f.write("   double s[%i];\n" % (m))
    f.write("   double y[%i];\n" % (p))
    f.write("   double z[%i];\n" % (m))
    f.write("   double L[%i];\n" % (Lnnz))
    f.write("   double D[%i];\n" % (n + m + p))
    f.write("   double W[%i];\n" % Wnnz)
    f.write("   double lambda[%i];\n" % m)
    f.write("   double Winv[%i];\n" % Wnnz)
    f.write("   double WtW[%i];\n" % Wnnz)
    f.write("   double kkt_rhs[%i];\n" % (n + m + p))
    f.write("   double kkt_res[%i];\n" % (n + m + p))
    f.write("   double xyz[%i];\n" % (n + m + p))
    f.write("   double xyzbuff[%i];\n" % (n + m + p))
    f.write("   double sbar[%i];\n" % max(q))
    f.write("   double zbar[%i];\n" % max(q))
    f.write("   double mu;\n\n")
    f.write("   double kkt_reg;\n")
    f.write("   double eps_gap;\n")
    f.write("   double eps_feas;\n")
    f.write("   unsigned char solved;\n")
    f.write("} Workspace;\n\n")
    f.write("#endif")
    f.close()

def generate_ldl(solver_dir, n, m, p, P, A, G, perm, Lidx, Wsparse2dense):
    f = open(solver_dir + "/ldl.h", "a")
    f.write("#ifndef LDL_H\n")
    f.write("#define LDL_H\n\n")
    f.write("#include \"workspace.h\"\n\n")

    f.write("void ldl(Workspace* work);\n")
    f.write("void tri_solve(Workspace* work);\n")
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
        write_Kelem(f, j, j, n, m, p, P, A, G, perm, Wsparse2dense, True)
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
                write_Kelem(f, j, i, n, m, p, P, A, G, perm, Wsparse2dense, True)
                for k in range(j):
                    if (Lidx[k * N + i] and Lidx[k * N + j]):
                        f.write(" - work->L[%i] * work->L[%i] * work->D[%i]" % (Lsparse2dense[k * N + i], Lsparse2dense[k * N + j], k))
                f.write(";\n")
                f.write("   work->L[%i] /= work->D[%i];\n" % (Lnnz, j))
                Lnnz += 1
    f.write("}\n\n")

    f.write("void tri_solve(Workspace* work){\n")
    for i in range(N):
        f.write("   work->xyzbuff[%i] = work->kkt_rhs[%i]" % (i, perm[i]))
        for j in range(i):
            if (Lidx[j * N + i]):
                f.write(" - work->L[%i] * work->xyzbuff[%i]" % (Lsparse2dense[j * N + i], j))
        f.write(";\n")

    for i in range(N):
        f.write("   work->xyzbuff[%i] /= work->D[%i];\n" % (i, i))

    for i in range(N-1, -1, -1):
        f.write("   work->xyz[%i] = work->xyzbuff[%i]" % (perm[i], i))
        for j in range(i+1, N):
            if (Lidx[i * N + j]):
                f.write(" - work->L[%i] * work->xyz[%i]" % (Lsparse2dense[i * N + j], perm[j]))
        f.write(";\n")
    f.write("}")

    f.close()
    return Lsparse2dense

def generate_cone(solver_dir, m, Wnnz, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/cone.h", "a")
    f.write("#ifndef CONE_H\n")
    f.write("#define CONE_H\n\n")
    f.write("#include \"utils.h\"\n\n")

    f.write("double soc_residual(double* u, int n);\n")
    f.write("double soc_residual2(double* u, int n);\n")
    f.write("double cone_residual(double* u, int l, int nsoc, int* q);\n")
    f.write("void bring2cone(double* u, int l, int nsoc, int* q);\n")
    f.write("void compute_mu(Workspace* work);\n")
    f.write("void compute_nt_scaling(Workspace* work);\n")
    f.write("void compute_lambda(Workspace* work);\n")
    f.write("void compute_WtW(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    f = open(solver_dir + "/cone.c", "a")
    f.write("#include \"cone.h\"\n\n")
    f.write("double soc_residual(double* u, int n){\n")
    f.write("   double res = 0;\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res += u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   res = qcos_sqrt(res) - u[0];\n")
    f.write("   return res;\n")
    f.write("}\n\n")
    f.write("double soc_residual2(double* u, int n){\n")
    f.write("   double res = u[0] * u[0];\n")
    f.write("   for (int i = 1; i < n; ++i) {\n")
    f.write("      res -= u[i] * u[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")

    f.write("double cone_residual(double* u, int l, int nsoc, int* q){\n")
    f.write("   double res = -1e7;\n")
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < l; ++idx) {\n")
    f.write("      res = qcos_max(-u[idx], res);\n")
    f.write("   }\n")
    f.write("   for (int i = 0; i < nsoc; ++i) {\n")
    f.write("      res = qcos_max(soc_residual(&u[idx], q[i]), res);\n")
    f.write("      idx += q[i];\n")
    f.write("   }\n")
    f.write("   return res;\n")
    f.write("}\n\n")
    
    f.write("void bring2cone(double* u, int l, int nsoc, int* q){\n")
    f.write("   if (cone_residual(u, l, nsoc, q) >= 0) {\n")
    f.write("      double a = 0.0;\n\n")
    f.write("      int idx;\n")
    f.write("      for (idx = 0; idx < l; ++idx) {\n")
    f.write("         a = qcos_max(a, -u[idx]);\n")
    f.write("      }\n")
    f.write("      a = qcos_max(a, 0.0);\n\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         double soc_res = soc_residual(&u[idx], q[i]);\n")
    f.write("         if (soc_res > 0 && soc_res > a) {\n")
    f.write("            a = soc_res;\n")
    f.write("         }\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("      for (idx = 0; idx < l; ++idx){\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("      }\n")
    f.write("      for (int i = 0; i < nsoc; ++i) {\n")
    f.write("         u[idx] += (1 + a);\n")
    f.write("         idx += q[i];\n")
    f.write("      }\n")
    f.write("   }\n")
    f.write("}\n")

    f.write("void compute_mu(Workspace* work){\n")
    if (m == 0):
        f.write("   work->mu = 0.0;\n")
    else:
        f.write("   work->mu = (dot(work->s, work->z, work->m) / work->m);\n")
    f.write("}\n\n")

    f.write("void compute_nt_scaling(Workspace* work){\n")
    for i in range(Wnnz):
        f.write("   work->W[%i] = 0.0;\n" % i)
        f.write("   work->Winv[%i] = 0.0;\n" % i)
    f.write("   int idx;\n")
    f.write("   for (idx = 0; idx < work->l; ++idx){\n")
    f.write("       work->W[idx] = qcos_sqrt(safe_div(work->s[idx], work->z[idx]));\n")
    f.write("       work->Winv[idx] = safe_div(1.0, work->W[idx]);\n")
    f.write("   }\n\n")

    f.write("   int nt_idx = idx;\n")
    f.write("   for (int i = 0; i < work->nsoc; ++i) {\n")
    f.write("       // Compute normalized vectors.\n")
    f.write("       double s_scal = soc_residual2(&work->s[idx], work->q[i]);\n")
    f.write("       s_scal = qcos_sqrt(s_scal);\n")
    f.write("       double f = safe_div(1.0, s_scal);\n")
    f.write("       scale_arrayf(&work->s[idx], work->sbar, f, work->q[i]);\n\n")

    f.write("       double z_scal = soc_residual2(&work->z[idx], work->q[i]);\n")
    f.write("       z_scal = qcos_sqrt(z_scal);\n")
    f.write("       f = safe_div(1.0, z_scal);\n")
    f.write("       scale_arrayf(&work->z[idx], work->zbar, f, work->q[i]);\n\n")

    f.write("       double gamma = qcos_sqrt(0.5 * (1 + dot(work->sbar, work->zbar, work->q[i])));\n")
    f.write("       f = safe_div(1.0, (2 * gamma));\n\n")
    f.write("       // Overwrite sbar with wbar.\n")
    f.write("       work->sbar[0] = f * (work->sbar[0] + work->zbar[0]);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j){\n")
    f.write("           work->sbar[j] = f * (work->sbar[j] - work->zbar[j]);\n")
    f.write("       }\n\n")
    f.write("       // Overwrite zbar with v.\n")
    f.write("       f = safe_div(1.0, qcos_sqrt(2 * (work->sbar[0] + 1)));\n")
    f.write("       work->zbar[0] = f * (work->sbar[0] + 1.0);\n")
    f.write("       for (int j = 1; j < work->q[i]; ++j) {\n")
    f.write("           work->zbar[j] = f * work->sbar[j];\n")
    f.write("       }\n\n")
    f.write("       // Compute W for second-order cones.\n")
    f.write("       int shift = 0;\n")
    f.write("       f = qcos_sqrt(safe_div(s_scal, z_scal));\n")
    f.write("       double finv = safe_div(1.0, f);\n")
    f.write("       for (int j = 0; j < work->q[i]; ++j) {\n")
    f.write("           for (int k = 0; k <= j; ++k) {\n")
    f.write("               work->W[nt_idx + shift] = 2 * (work->zbar[k] * work->zbar[j]);\n")
    f.write("               if (j != 0 && k == 0) {\n")
    f.write("                   work->Winv[nt_idx + shift] = -work->W[nt_idx + shift];\n")
    f.write("               }\n")
    f.write("               else {\n")
    f.write("                   work->Winv[nt_idx + shift] = work->W[nt_idx + shift];\n")
    f.write("               }\n")
    f.write("               if (j == k && j == 0) {\n")
    f.write("                   work->W[nt_idx + shift] -= 1;\n")
    f.write("                   work->Winv[nt_idx + shift] -= 1;\n")
    f.write("               }\n")
    f.write("               else if (j == k) {\n")
    f.write("                   work->W[nt_idx + shift] += 1;\n")
    f.write("                   work->Winv[nt_idx + shift] += 1;\n")
    f.write("               }\n")
    f.write("               work->W[nt_idx + shift] *= f;\n")
    f.write("               work->Winv[nt_idx + shift] *= finv;\n")
    f.write("               shift += 1;\n")
    f.write("           }\n")
    f.write("       }\n")
    f.write("       idx += work->q[i];\n")
    f.write("       nt_idx += (work->q[i] * work->q[i] + work->q[i]) / 2;\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void compute_WtW(Workspace* work){\n")
    for i in range(m):
        for j in range(i, m):
            if (Wsparse2dense[j * m + i] != -1):
                f.write("   work->WtW[%i] = " % Wsparse2dense[j * m + i])
                for k in range(m):
                    row1 = k
                    col1 = j
                    row2 = k
                    col2 = i
                    if (col1 < row1):
                        row1, col1 = col1, row1
                    if (col2 < row2):
                        row2, col2 = col2, row2
                    if (Wsparse2dense[col1 * m + row1] != -1 and Wsparse2dense[col2 * m + row2] != -1):
                        f.write(" + work->W[%i] * work->W[%i]" % (Wsparse2dense[col1 * m + row1], Wsparse2dense[col2 * m + row2]))
                f.write(";\n")
    f.write("}\n\n")

    f.write("void compute_lambda(Workspace* work) {\n")
    for i in range(m):
        f.write("   work->lambda[%i] = " % i)
        for j in range(m):
            row = i
            col = j
            if (col < row):
                row, col = col, row
            if (Wsparse2dense[col * m + row] != -1):
                f.write(" + work->W[%i] * work->z[%i]" % (Wsparse2dense[col * m + row], j))
        f.write(";\n")
    f.write("}\n\n")
    f.close()

def generate_kkt(solver_dir, n, m, p, P, c, A, b, G, h, perm, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/kkt.h", "a")
    f.write("#ifndef KKT_H\n")
    f.write("#define KKT_H\n\n")
    f.write("#include \"workspace.h\"\n\n")
    f.write("void compute_kkt_residual(Workspace* work);\n")
    f.write("#endif")
    f.close()

    # Write source.
    N = n + m + p
    f = open(solver_dir + "/kkt.c", "a")
    f.write("#include \"kkt.h\"\n\n")
    f.write("void compute_kkt_residual(Workspace* work){\n")

    f.write("   // Zero out NT Block.\n")
    for i in range(m**2):
        if (Wsparse2dense[i] != -1):
            f.write("   work->WtW[%i] = 0.0;\n" % Wsparse2dense[i])

    for i in range(N):
        f.write("   work->kkt_res[%i] = " % i)
        for j in range(N):
            if (write_Kelem(f, i, j, n, m, p, P, A, G, np.linspace(0, N-1, N, dtype=np.int32), Wsparse2dense, False)):
                if (j < n):
                    f.write(" * work->x[%i]" % j)
                elif (j >= n and j < n + p):
                    f.write(" * work->y[%i]" % (j - n))
                elif (j >= n + p and j < n + m + p):
                    f.write(" * work->z[%i]" % (j - n - p))
                f.write(" + ")
        
        # Add [c;-b;s-h]
        if (i < n):
            f.write(" work->c[%i]" % i)
        elif (i >= n and i < n + p):
            f.write(" - work->b[%i]" % (i - n))
        elif (i >= n + p and i < n + m + p):
            f.write("work->s[%i] - work->h[%i]" % (i - n - p, i - n - p))
        else:
            raise ValueError("Should not happen.")
        f.write(";\n")
    f.write("}")
    f.close()

def generate_utils(solver_dir, n, m, p, P, c, A, b, G, h, l, nsoc, q, Wsparse2dense):
    # Write header.
    f = open(solver_dir + "/utils.h", "a")
    f.write("#ifndef UTILS_H\n")
    f.write("#define UTILS_H\n\n")
    f.write("#include \"workspace.h\"\n\n")
    f.write("#define qcos_abs(x) ((x)<0 ? -(x) : (x))\n")
    f.write("#define safe_div(a, b) (qcos_abs(a) > 1e-15) ? (a / b) : 1e16\n")
    f.write("#include <math.h>\n")
    f.write("#define qcos_sqrt(a) sqrt(a)\n")
    f.write("#define qcos_max(a, b) (((a) > (b)) ? (a) : (b))\n\n")
    f.write("void load_data(Workspace* work);\n")
    f.write("void copy_arrayf(double* x, double* y, int n);\n")
    f.write("void copy_and_negate_arrayf(double* x, double* y, int n);\n")
    f.write("double dot(double* x, double* y, int n);\n")
    f.write("void scale_arrayf(double* x, double* y, double s, int n);\n")
    f.write("unsigned char check_stopping(Workspace* work);\n")
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
   
    f.write("   work->mu = 0.0;\n")
    f.write("   work->kkt_reg = 1e-7;\n")
    f.write("   work->eps_gap = 1e-7;\n")
    f.write("   work->eps_feas = 1e-7;\n")
    f.write("   work->solved = 0;\n")
    f.write("\n")
    f.write("}\n\n")

    f.write("void copy_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("void copy_and_negate_arrayf(double* x, double* y, int n) {\n")
    f.write("   for (int i = 0; i < n; ++i) {\n")
    f.write("      y[i] = -x[i];\n")
    f.write("   }\n")
    f.write("}\n\n")

    f.write("double dot(double* x, double* y, int n){\n")
    f.write("   double ans = 0;\n")
    f.write("      for (int i = 0; i < n; ++i){\n")
    f.write("         ans += x[i] * y[i];\n")
    f.write("      }\n")
    f.write("   return ans;\n")
    f.write("}\n\n")

    f.write("void scale_arrayf(double* x, double* y, double s, int n){\n")
    f.write("      for (int i = 0; i < n; ++i){\n")
    f.write("         y[i] = s * x[i];\n")
    f.write("      }\n")
    f.write("}\n\n")

    f.write("unsigned char check_stopping(Workspace* work){\n")
    f.write("   double res = 1e3;\n")
    f.write("   for (int i = 0; i < work->n + work->m + work->p; ++i){\n")
    f.write("      res = qcos_max(res, qcos_abs(work->kkt_res[i]));\n")
    f.write("   }\n")
    f.write("   if (res < work->eps_feas && (work->mu * work->m) < work->eps_gap) {\n")
    f.write("      return 1;\n")
    f.write("   }\n")
    f.write("   return 0;\n")
    f.write("}\n")
    f.close()

def generate_solver(solver_dir, m, Wsparse2dense):
    f = open(solver_dir + "/qcos_custom.h", "a")
    f.write("#ifndef QCOS_CUSTOM_H\n")
    f.write("#define QCOS_CUSTOM_H\n\n")
    f.write("#include \"cone.h\"\n")
    f.write("#include \"kkt.h\"\n")
    f.write("#include \"ldl.h\"\n")
    f.write("#include \"utils.h\"\n")
    f.write("#include \"workspace.h\"\n\n")
    f.write("void qcos_custom_solve(Workspace* work);\n")
    f.write("#endif")
    f.close()

    f = open(solver_dir + "/qcos_custom.c", "a")
    f.write("#include \"qcos_custom.h\"\n\n")
    f.write("void initialize_ipm(Workspace* work){\n")
    f.write("   // Set NT block to -I.\n")
    for i in range(m**2):
        if (Wsparse2dense[i] != -1):
            f.write("   work->WtW[%i] = 0.0;\n" % Wsparse2dense[i])
    for i in range(m):
        f.write("   work->WtW[%i] = -1.0;\n" % Wsparse2dense[i * m + i])
    
    f.write("\n   // kkt_rhs = [-c;b;h].\n")
    f.write("   for(int i = 0; i < work->n; ++i){\n")
    f.write("       work->kkt_rhs[i] = -work->c[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->p; ++i){\n")
    f.write("       work->kkt_rhs[work->n + i] = work->b[i];\n")
    f.write("   }\n")
    f.write("   for(int i = 0; i < work->m; ++i){\n")
    f.write("       work->kkt_rhs[work->n + work->p + i] = work->h[i];\n")
    f.write("   }\n\n")
    f.write("   ldl(work);\n")
    f.write("   tri_solve(work);\n")
    f.write("   copy_arrayf(work->xyz, work->x, work->n);\n")
    f.write("   copy_arrayf(&work->xyz[work->n], work->y, work->p);\n")
    f.write("   copy_arrayf(&work->xyz[work->n + work->p], work->z, work->m);\n")
    f.write("   copy_and_negate_arrayf(&work->xyz[work->n + work->p], work->s, work->m);\n")
    f.write("   bring2cone(work->s, work->l, work->nsoc, work->q);\n")
    f.write("   bring2cone(work->z, work->l, work->nsoc, work->q);\n")
    f.write("}\n\n")

    f.write("void qcos_custom_solve(Workspace* work){\n")
    f.write("   initialize_ipm(work);\n")
    f.write("   for (int i = 1; i < 2; ++i) {\n")
    f.write("      compute_kkt_residual(work);\n")
    f.write("      compute_mu(work);\n")
    f.write("      if (check_stopping(work)) {\n")
    f.write("         work->solved = 1;\n")
    f.write("         return;\n")
    f.write("      }\n")
    f.write("      compute_nt_scaling(work);\n")
    f.write("      compute_lambda(work);\n")
    f.write("      compute_WtW(work);\n")
    f.write("   }\n")
    f.write("}\n\n")
    f.close()

def generate_runtest(solver_dir, P, c, A, b, G, h, l, nsoc, q):
    f = open(solver_dir + "/runtest.c", "a")
    f.write("#include <stdio.h>\n")
    f.write("#include \"qcos_custom.h\"\n\n")
    f.write("int main(){\n")
    f.write("   Workspace work;\n")
    f.write("   load_data(&work);\n")
    f.write("   qcos_custom_solve(&work);\n")
    f.write("   printf(\"xyz: {\");")
    f.write("   for(int i = 0; i < work.n + work.m + work.p; ++i){\n")
    f.write("   printf(\"%f, \", work.xyz[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\nkkt_res: {\");")
    f.write("   for(int i = 0; i < work.n + work.m + work.p; ++i){\n")
    f.write("   printf(\"%f, \", work.kkt_res[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"mu: %f\", work.mu);\n")

    f.write("   printf(\"\\nx: {\");")
    f.write("   for(int i = 0; i < work.n; ++i){\n")
    f.write("   printf(\"%f, \", work.x[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\ns: {\");")
    f.write("   for(int i = 0; i < work.m; ++i){\n")
    f.write("   printf(\"%f, \", work.s[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\ny: {\");")
    f.write("   for(int i = 0; i < work.p; ++i){\n")
    f.write("   printf(\"%f, \", work.y[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\nz: {\");")
    f.write("   for(int i = 0; i < work.m; ++i){\n")
    f.write("   printf(\"%f, \", work.z[i]);\n")
    f.write("   }\n")
    f.write("   printf(\"}\\n\");\n")

    f.write("   printf(\"\\nW: {\");")
    f.write("   for(int i = 0; i < 9; ++i){\n")
    f.write("   printf(\"%f, \", work.W[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\nWinv: {\");")
    f.write("   for(int i = 0; i < 9; ++i){\n")
    f.write("   printf(\"%f, \", work.Winv[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\nWtW: {\");")
    f.write("   for(int i = 0; i < 9; ++i){\n")
    f.write("   printf(\"%f, \", work.WtW[i]);\n")
    f.write("   }\n")

    f.write("   printf(\"\\nlambda: {\");")
    f.write("   for(int i = 0; i < work.m; ++i){\n")
    f.write("   printf(\"%f, \", work.lambda[i]);\n")
    f.write("   }\n")

    f.write("}")

    f.close()
